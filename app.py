#!/usr/bin/env python3
"""Учебный поиск: OpenAI embeddings + Pinecone, без генеративного ответа."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
NAME = os.getenv('PINECONE_INDEX', 'vpg02-ppe')
NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'vpg02-ppe')
DIMENSION = 1536


def clients():
    missing = [k for k in ('OPENAI_API_KEY', 'PINECONE_API_KEY') if not os.getenv(k)]
    if missing:
        raise ValueError('Заполните в .env: ' + ', '.join(missing))
    options = {'api_key': os.environ['OPENAI_API_KEY']}
    if os.getenv('OPENAI_BASE_URL'):
        options['base_url'] = os.environ['OPENAI_BASE_URL']
    return OpenAI(**options), Pinecone(api_key=os.environ['PINECONE_API_KEY'])


def embed(client, texts):
    result = client.embeddings.create(model=MODEL, input=texts)
    vectors = [item.embedding for item in sorted(result.data, key=lambda x: x.index)]
    if len(vectors) != len(texts) or any(len(v) != DIMENSION for v in vectors):
        raise ValueError(f'Ожидались эмбеддинги размерности {DIMENSION}; проверьте модель.')
    return vectors


def index_for(pc, create=False):
    if not pc.has_index(NAME):
        if not create:
            raise ValueError(f'Индекс {NAME} не найден. Сначала выполните setup.')
        pc.create_index(name=NAME, dimension=DIMENSION, metric='cosine', spec=ServerlessSpec(
            cloud=os.getenv('PINECONE_CLOUD', 'aws'), region=os.getenv('PINECONE_REGION', 'us-east-1')))
        print(f'Создан индекс {NAME}; ожидаем готовности...')
    deadline = time.monotonic() + 120
    while True:
        info = pc.describe_index(NAME)
        if info.dimension != DIMENSION or info.metric != 'cosine':
            raise ValueError('Существующий индекс не соответствует размерности 1536 или метрике cosine.')
        if info.status['ready']:
            return pc.Index(host=info.host)
        if time.monotonic() >= deadline:
            raise TimeoutError('Индекс ещё не готов. Повторите команду позже.')
        time.sleep(2)


def search(index, vector, top_k=3):
    response = index.query(namespace=NAMESPACE, vector=vector, top_k=top_k, include_metadata=True)
    return response.matches


def show(query, matches):
    print(f'\nЗапрос: {query}')
    if not matches:
        print('  Совпадений нет.')
    for n, item in enumerate(matches, 1):
        print(f"  {n}. ID: {item.id} | сходство: {item.score:.4f}\n     Текст: {(item.metadata or {}).get('text', '[текст отсутствует]')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup', help='создать индекс при необходимости и загрузить 24 записи')
    sub.add_parser('demo', help='выполнить все запросы из queries.json')
    p = sub.add_parser('search', help='выполнить свой запрос')
    p.add_argument('text')
    args = parser.parse_args()
    ai, pc = clients()
    if args.command == 'setup':
        data = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
        vectors = embed(ai, [x['text'] for x in data])
        print(f'Созданы эмбеддинги: {len(vectors)}, размерность: {len(vectors[0])}, модель: {MODEL}')
        idx = index_for(pc, create=True)
        idx.upsert(namespace=NAMESPACE, vectors=[{'id': d['id'], 'values': v, 'metadata': {'text': d['text']}} for d, v in zip(data, vectors)])
        print(f'Отправлено записей: {len(data)}; индекс: {NAME}; пространство: {NAMESPACE}')
        print('Pinecone индексирует записи не мгновенно; при необходимости повторите demo через несколько секунд.')
    else:
        idx = index_for(pc)
        queries = json.loads((ROOT / 'queries.json').read_text(encoding='utf-8')) if args.command == 'demo' else [{'query': args.text}]
        vectors = embed(ai, [q['query'] for q in queries])
        for q, vector in zip(queries, vectors):
            matches = search(idx, vector)
            show(q['query'], matches)



if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        sys.exit(1)
