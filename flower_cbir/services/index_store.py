"""Atomic publication of raw features, normalization, vectors and provenance."""
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from psycopg import sql
from psycopg.types.json import Jsonb

from flower_cbir.core.contract import file_sha256, fingerprint, verify_contract
from flower_cbir.core.schema import feature_columns, validate_features
from flower_cbir.settings import PROJECT_ROOT


def backup_database(repository):
    snapshot = {'format': 'cbir-postgres-snapshot-v1', 'schema': repository.schema, 'tables': {}}
    names = ['flower_images', 'flower_masks', 'flower_features', 'feature_normalization', 'normalized_features',
             'retrieval_history', 'cbir_index_versions', 'cbir_index_entries']
    with repository.conn.transaction():
        repository.conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        for name in names:
            exists = repository.conn.execute('SELECT to_regclass(%s)', (repository.schema+'.'+name,)).fetchone()[0]
            if exists:
                rows = repository.conn.execute(sql.SQL('SELECT row_to_json(t) FROM {} t').format(sql.Identifier(name))).fetchall()
                snapshot['tables'][name] = [row[0] for row in rows]
        snapshot['columns'] = repository.conn.execute('''
            SELECT table_name,column_name,data_type,is_identity,identity_generation,column_default
            FROM information_schema.columns WHERE table_schema=%s ORDER BY table_name,ordinal_position
        ''', (repository.schema,)).fetchall()
        snapshot['database'] = repository.conn.execute('SELECT current_database()').fetchone()[0]
    directory = PROJECT_ROOT/'output/backups'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-postgres.json.gz')
    with gzip.open(destination, 'wt', encoding='utf-8') as stream:
        json.dump(snapshot, stream, ensure_ascii=False, separators=(',', ':'))
    return destination


def ensure_schema(connection):
    connection.execute('''
        CREATE TABLE IF NOT EXISTS cbir_index_versions (
            version_id TEXT PRIMARY KEY,
            normalization_id INTEGER NOT NULL UNIQUE REFERENCES feature_normalization(normalization_id),
            contract JSONB NOT NULL, manifest JSONB NOT NULL, normalizer JSONB NOT NULL,
            active BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS cbir_one_active_index ON cbir_index_versions ((active)) WHERE active')
    connection.execute('''
        CREATE TABLE IF NOT EXISTS cbir_index_entries (
            version_id TEXT NOT NULL REFERENCES cbir_index_versions(version_id),
            image_id INTEGER NOT NULL REFERENCES flower_images(image_id),
            raw_vector DOUBLE PRECISION[] NOT NULL CHECK(array_length(raw_vector,1)=163),
            normalized_vector DOUBLE PRECISION[] NOT NULL CHECK(array_length(normalized_vector,1)=163),
            metadata JSONB NOT NULL,
            PRIMARY KEY(version_id,image_id)
        )
    ''')
    connection.execute('ALTER TABLE retrieval_history ADD COLUMN IF NOT EXISTS index_version TEXT')
    connection.execute('ALTER TABLE retrieval_history ADD COLUMN IF NOT EXISTS weights DOUBLE PRECISION[]')
    connection.execute('ALTER TABLE retrieval_history ADD COLUMN IF NOT EXISTS query_context JSONB')


def _relative(path):
    import os
    return Path(os.path.relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()


def publish_index(repository, data, *, replace_all=False, backup_artifact=None):
    contract = verify_contract(data['contract'])
    raw = np.asarray(data['raw_features'], dtype=np.float32)
    normalized = np.asarray(data['normalized_features'], dtype=np.float32)
    validate_features(raw)
    metadata = data['metadata']
    if raw.shape != (len(metadata),163) or not np.array_equal(data['normalizer'].transform(raw),normalized):
        raise ValueError('Cannot publish inconsistent raw features and normalization.')
    if len({m['filename'] for m in metadata}) != len(metadata):
        raise ValueError('Duplicate filenames in the publication batch.')
    for item in metadata:
        for key in ['image', 'mask']:
            if file_sha256(item[key+'_path']) != item[key+'_sha256']:
                raise ValueError(f'Source changed since extraction: {item["filename"]}, {key}')
        if item.get('mask_provenance') not in ('deeplabv3', 'reviewed_final_segmentation_v2',
                                                'deeplabv3plus_v2_postprocessed'):
            raise ValueError('Only verified model masks or reviewed v2 ground truth may be published.')
    if not replace_all and repository.active_version() == data['version_id']:
        existing = repository.load_index()
        by_name = {m['filename']:i for i,m in enumerate(existing['metadata'])}
        if len(by_name) != len(metadata) or any(
            not np.array_equal(raw[i],existing['raw_features'][by_name[m['filename']]])
            for i,m in enumerate(metadata)):
            raise ValueError('The active immutable version differs from the publication batch.')
        return {'version_id':data['version_id'],'already_active':True,'rows':len(metadata)}
    if replace_all:
        backup = Path(backup_artifact).resolve() if backup_artifact is not None else None
        if backup is None or not backup.is_file() or backup.stat().st_size == 0:
            raise ValueError('A validated pg_dump backup is required before replacing all tables.')
    else:
        backup = backup_database(repository)
    connection = repository.conn
    with repository.lock, connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('cbir_publish_index'))")
        ensure_schema(connection)
        if replace_all:
            connection.execute('''TRUNCATE TABLE retrieval_history,cbir_index_entries,
                cbir_index_versions,normalized_features,flower_features,flower_masks,
                flower_images,feature_normalization RESTART IDENTITY CASCADE''')
        connection.execute('LOCK TABLE flower_images,flower_features,feature_normalization,normalized_features IN SHARE ROW EXCLUSIVE MODE')
        existing_names = {row[0] for row in connection.execute('SELECT filename FROM flower_images').fetchall()}
        if existing_names - {m['filename'] for m in metadata}:
            raise ValueError('Publication must include every existing gallery image; partial replacement is rejected.')
        existing = connection.execute('SELECT normalization_id,manifest FROM cbir_index_versions WHERE version_id=%s',
                                      (data['version_id'],)).fetchone()
        if existing:
            normalization_id, previous_manifest = existing
            if previous_manifest['fingerprint'] != data['manifest']['fingerprint']:
                raise ValueError('An immutable version ID was reused for different data.')
        else:
            normalization_id = connection.execute('''
                INSERT INTO feature_normalization (method,feature_means,feature_stds)
                VALUES (%s,%s,%s) RETURNING normalization_id
            ''', ('report1_f2f4_zscore',data['normalizer'].mean.tolist(),data['normalizer'].std.tolist())).fetchone()[0]
            connection.execute('''
                INSERT INTO cbir_index_versions (version_id,normalization_id,contract,manifest,normalizer)
                VALUES (%s,%s,%s,%s,%s)
            ''',(data['version_id'],normalization_id,Jsonb(contract),Jsonb(data['manifest']),Jsonb(data['normalizer'].to_dict())))
        for index,item in enumerate(metadata):
            identity = connection.execute('''
                INSERT INTO flower_images (filename,image_path,mask_path,width,height,image_format,image_data)
                VALUES (%s,%s,%s,384,384,%s,%s)
                ON CONFLICT (filename) DO UPDATE SET image_path=EXCLUDED.image_path,
                    mask_path=EXCLUDED.mask_path,width=384,height=384,image_format=EXCLUDED.image_format,
                    image_data=EXCLUDED.image_data
                RETURNING image_id
            ''',(item['filename'],_relative(item['image_path']),_relative(item['mask_path']),
                 Path(item['image_path']).suffix.lstrip('.').upper(),
                 Path(item['image_path']).read_bytes())).fetchone()[0]
            connection.execute('''
                INSERT INTO flower_features (image_id,hs_histogram,v_histogram,lab_mean,lab_std,lbp_histogram,
                    circularity,solidity,aspect_ratio,rotated_extent,radial_spectrum,iap,iep)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (image_id) DO UPDATE SET hs_histogram=EXCLUDED.hs_histogram,
                    v_histogram=EXCLUDED.v_histogram,lab_mean=EXCLUDED.lab_mean,lab_std=EXCLUDED.lab_std,
                    lbp_histogram=EXCLUDED.lbp_histogram,circularity=EXCLUDED.circularity,solidity=EXCLUDED.solidity,
                    aspect_ratio=EXCLUDED.aspect_ratio,rotated_extent=EXCLUDED.rotated_extent,
                    radial_spectrum=EXCLUDED.radial_spectrum,iap=EXCLUDED.iap,iep=EXCLUDED.iep,extracted_at=CURRENT_TIMESTAMP
            ''',(identity,*feature_columns(raw[index])))
            connection.execute('''
                INSERT INTO flower_masks (image_id,mask_data,mask_format,mask_sha256,source_version)
                VALUES (%s,%s,'PNG',%s,%s)
                ON CONFLICT (image_id) DO UPDATE SET mask_data=EXCLUDED.mask_data,
                    mask_format=EXCLUDED.mask_format,mask_sha256=EXCLUDED.mask_sha256,
                    source_version=EXCLUDED.source_version,updated_at=CURRENT_TIMESTAMP
            ''',(identity,Path(item['mask_path']).read_bytes(),item['mask_sha256'],
                 item['mask_provenance']))
            connection.execute('''
                INSERT INTO normalized_features (image_id,normalization_id,normalized_vector,pgvector_embedding)
                VALUES (%s,%s,%s,%s::vector(163))
                ON CONFLICT (image_id) DO UPDATE SET normalization_id=EXCLUDED.normalization_id,
                    normalized_vector=EXCLUDED.normalized_vector,
                    pgvector_embedding=EXCLUDED.pgvector_embedding,created_at=CURRENT_TIMESTAMP
            ''',(identity,normalization_id,normalized[index].tolist(),normalized[index].tolist()))
            record = {**item,'image_id':identity,'image_path':_relative(item['image_path']),
                      'mask_path':_relative(item['mask_path'])}
            connection.execute('''
                INSERT INTO cbir_index_entries (version_id,image_id,raw_vector,normalized_vector,metadata)
                VALUES (%s,%s,%s,%s,%s) ON CONFLICT (version_id,image_id) DO NOTHING
            ''',(data['version_id'],identity,raw[index].tolist(),normalized[index].tolist(),Jsonb(record)))
        count = connection.execute('SELECT count(*) FROM cbir_index_entries WHERE version_id=%s',
                                   (data['version_id'],)).fetchone()[0]
        if count != len(metadata):
            raise ValueError('Incomplete immutable version; refusing activation.')
        actual = connection.execute('''
            SELECT i.filename,e.raw_vector,e.normalized_vector,n.normalized_vector,
                   n.pgvector_embedding::real[]
            FROM cbir_index_entries e JOIN flower_images i ON i.image_id=e.image_id
            JOIN normalized_features n ON n.image_id=i.image_id AND n.normalization_id=%s
            WHERE e.version_id=%s
        ''',(normalization_id,data['version_id'])).fetchall()
        positions={m['filename']:i for i,m in enumerate(metadata)}
        if len(actual)!=len(metadata) or any(
            not np.array_equal(raw[positions[name]],np.asarray(rv,np.float32)) or
            not np.array_equal(normalized[positions[name]],np.asarray(nv,np.float32)) or
            not np.array_equal(normalized[positions[name]],np.asarray(legacy_nv,np.float32)) or
            not np.array_equal(normalized[positions[name]],np.asarray(pgvector_nv,np.float32))
            for name,rv,nv,legacy_nv,pgvector_nv in actual):
            raise ValueError('Database round-trip differs from the staged batch; activation refused.')
        # Check the contract again before committing a potentially long import.
        verify_contract(contract)
        connection.execute('UPDATE cbir_index_versions SET active=FALSE WHERE active')
        connection.execute('UPDATE cbir_index_versions SET active=TRUE WHERE version_id=%s',(data['version_id'],))
    return {'version_id':data['version_id'],'normalization_id':normalization_id,
            'rows':len(metadata),'backup':str(backup),'already_active':False}
