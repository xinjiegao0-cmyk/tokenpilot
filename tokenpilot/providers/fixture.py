"""Deterministic structured-record interpreter, exclusively for smoke testing."""
import json
from decimal import Decimal

from tokenpilot.providers.base import BaseProvider, ProviderResponse, Usage


class FixtureProvider(BaseProvider):
    name = 'fixture'
    simulation = True

    def describe(self):
        return {'provider': self.name, 'adapter': 'record-interpreter-v1',
                'usage_method': 'synthetic UTF-8 byte counts, not a model tokenizer'}

    def generate(self, request):
        payload = json.loads(request.prompt.split('\n', 1)[1])
        query, documents = payload['query'], payload['records']
        matches = [doc for doc in documents if doc['key'] == query['key']]
        answer, evidence = None, []
        if matches:
            if query['operation'] == 'lookup':
                answer, evidence = matches[0]['value'], [matches[0]['id']]
            elif query['operation'] == 'latest':
                latest = max(matches, key=lambda doc: doc['revision'])
                answer, evidence = latest['value'], [latest['id']]
            elif query['operation'] == 'sum':
                answer = sum(doc['value'] for doc in matches)
                evidence = [doc['id'] for doc in matches]
            elif query['operation'] == 'join':
                linked = [doc for doc in documents if doc['key'] == matches[0]['value']]
                if linked:
                    answer, evidence = linked[0]['value'], [matches[0]['id'], linked[0]['id']]
        text = json.dumps({'answer': answer, 'citations': evidence})
        usage = Usage(len(request.prompt.encode('utf-8')), len(text.encode('utf-8')))
        completed = usage.output_tokens <= request.max_output_tokens
        return ProviderResponse(text if completed else '', usage, Decimal(usage.total_tokens) / Decimal('1000000'),
                                'synthetic fixture rate: USD 1 per million byte-units',
                                'simulated', completed=completed, metadata={'model': 'record-interpreter-v1'})
