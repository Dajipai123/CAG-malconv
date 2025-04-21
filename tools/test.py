from elasticsearch import Elasticsearch

es = Elasticsearch(hosts='http://172.22.105.146:9200', request_timeout=3600)

settings = {
    "index.mapping.total_fields.limit": 10000
}

es.indices.put_settings(index='malware', body=settings)