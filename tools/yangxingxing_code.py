from elasticsearch import Elasticsearch


es = Elasticsearch(hosts='http://172.22.105.146:9200', request_timeout=3600)
# es = Elasticsearch([{'host': '172.22.105.146', 'port': 9200, 'scheme':'http'}], headers={'content-type': 'application/json'})
my_index='pdns'

def send_to_elasticsearch(index):
    try:
        res = es.search(index=index, body={"query": {"match_all": {}}, "size": 10})
        return res['hits']['hits']
    except Exception as e:
        print(f"Error getting messages: {e}")

def search_from_elasticsearch(index=my_index):
    try:
        return es.get(index=index, id=id)
    except UnicodeEncodeError as e:
        print(f"Error encoding data: {e}")

def get_first_10_records(index):
    es = Elasticsearch(hosts='http://172.22.105.146:9200', timeout=3600 ,headers={"Content-Type": "application/json"})
    try:
        res = es.search(index=index, body={"query": {"match_all": {}}, "size": 10})
        return res['hits']['hits']
    except Exception as e:
        print(f"Error getting records: {e}")



if __name__ == "__main__":
    json_data={
        "id": 0x123,
        "value": ['test','my','data'],
    }
    # send_to_elasticsearch(json_data["id"],json_data,index='pdns')

    messages = search_from_elasticsearch(index='6fd4849beabb6b6d40230e9f4d491d26 ')
    print(messages)

    # # 删除index中的所有数据
    # es.delete_by_query(index='6fd4849beabb6b6d40230e9f4d491d26', body={"query": {"match_all": {}}})