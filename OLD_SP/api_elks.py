from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from typing import List, Tuple, Dict, Any
import os
from es_module import *



# get the list of all the (distinct) protocols contained in elastic search with the index pcap-flows
def get_disctinct_protocols(index_name):
    client = create_or_get_es()
    query = {
        "size": 0,
        "aggs": {
            "distinct_protocols": {
                "terms": {
                    "field": "protocol"
                }
            }
        }
    }
    response = client.search(index=index_name, body=query)
    return response['aggregations']['distinct_protocols']['buckets']


# get the list of all the (distinct) applications contained in elastic search with the index pcap-flows
def get_disctinct_applications(index_name):
    client = create_or_get_es()
    query = {
        "size": 0,
        "aggs": {
            "distinct_apps": {
                "terms": {
                    "field": "application_name"
                }
            }
        }
    }
    response = client.search(index=index_name, body=query)
    return [bucket['key'] for bucket in response['aggregations']['distinct_apps']['buckets']]



def get_flows_for_protocol(index: str, protocol: str) -> List[Dict[str, Any]]:
    query = {
        "query": {
            "term": {
                "protocol.keyword": protocol
            }
        }
    }
    response = es.search(index=index, body=query)
    return [hit['_source'] for hit in response['hits']['hits']]

def get_number_of_flows_for_protocol(index: str) -> List[Tuple[str, int]]:
    distinct_protocols = get_protocols(index)
    res = []
    for protocol in distinct_protocols:
        flows = get_flows_for_protocol(index, protocol)
        res.append((protocol, len(flows)))
    return res

def get_src_dst_size_per_protocol(index: str) -> List[Tuple[str, int, int]]:
    protocols = get_protocols(index)
    res = []
    for protocol in protocols:
        flows = get_flows_for_protocol(index, protocol)
        src_size = sum(flow['src2dst_bytes'] for flow in flows)
        dst_size = sum(flow['dst2src_bytes'] for flow in flows)
        res.append((protocol, src_size, dst_size))
    return res

def get_total_src_dst_bytes_per_protocol(index: str) -> List[Tuple[str, int, int]]:
    distinct_protocols = get_protocols(index)
    res = []
    for protocol in distinct_protocols:
        query = {
            "query": {
                "term": {
                    "protocol.keyword": protocol
                }
            },
            "aggs": {
                "total_src_bytes": {
                    "sum": {
                        "field": "src2dst_bytes"
                    }
                },
                "total_dst_bytes": {
                    "sum": {
                        "field": "dst2src_bytes"
                    }
                }
            }
        }
        response = es.search(index=index, body=query)
        src_bytes = response['aggregations']['total_src_bytes']['value'] or 0
        dst_bytes = response['aggregations']['total_dst_bytes']['value'] or 0
        res.append((protocol, src_bytes, dst_bytes))
    return res

def get_total_src_dst_packets_per_protocol(index: str) -> List[Tuple[str, int]]:
    distinct_protocols = get_protocols(index)
    res = []
    for protocol in distinct_protocols:
        query = {
            "query": {
                "term": {
                    "protocol.keyword": protocol
                }
            },
            "aggs": {
                "total_packets": {
                    "sum": {
                        "field": "src2dst_packets"
                    }
                }
            }
        }
        response = es.search(index=index, body=query)
        packets = response['aggregations']['total_packets']['value'] or 0
        res.append((protocol, packets))
    return res

def get_flows_for_application(index: str, application: str) -> List[Dict[str, Any]]:
    query = {
        "query": {
            "term": {
                "application_name.keyword": application
            }
        }
    }
    response = es.search(index=index, body=query)
    return [hit['_source'] for hit in response['hits']['hits']]

def get_number_of_flows_for_application(index: str) -> List[Tuple[str, int]]:
    distinct_applications = get_apps_list(index)
    res = []
    for application in distinct_applications:
        flows = get_flows_for_application(index, application)
        res.append((application, len(flows)))
    return res

def get_src_dst_size_per_application(index: str) -> List[Tuple[str, int, int]]:
    apps = get_apps_list(index)
    res = []
    for app in apps:
        flows = get_flows_for_application(index, app)
        src_size = sum(flow['src2dst_bytes'] for flow in flows)
        dst_size = sum(flow['dst2src_bytes'] for flow in flows)
        res.append((app, src_size, dst_size))
    return res

def get_total_bytes_per_application(index: str) -> List[Tuple[str, int]]:
    res = []
    src_dst = get_src_dst_size_per_application(index)
    for app in src_dst:
        res.append((app[0], app[1] + app[2]))
    return res

def get_total_packets_per_application(index: str) -> List[Tuple[str, int]]:
    apps = get_apps_list(index)
    res = []
    for app in apps:
        query = {
            "query": {
                "term": {
                    "application_name.keyword": app
                }
            },
            "aggs": {
                "total_packets": {
                    "sum": {
                        "field": "src2dst_packets"
                    }
                }
            }
        }
        response = es.search(index=index, body=query)
        packets = response['aggregations']['total_packets']['value'] or 0
        res.append((app, packets))
    return res
