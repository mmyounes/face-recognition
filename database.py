import os
from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

class CouchbaseManager:
    _cluster = None
    _bucket = None

    @classmethod
    def get_cluster(cls):
        if cls._cluster is None:
            auth = PasswordAuthenticator(
                os.getenv("COUCHBASE_USER"), 
                os.getenv("COUCHBASE_PASSWORD")
            )
            options = ClusterOptions(auth)
            
            # Apply the WAN development profile for high-latency environments
            options.apply_profile('wan_development')
            
            cls._cluster = Cluster(
                os.getenv("COUCHBASE_HOST"), 
                options
            )
            cls._cluster.wait_until_ready(timedelta(seconds=5))
            print("Connected to Couchbase ✅")
        return cls._cluster

    @classmethod
    def get_bucket(cls):
        if cls._bucket is None:
            cluster = cls.get_cluster()
            bucket_name = os.getenv("COUCHBASE_BUCKET")
            cls._bucket = cluster.bucket(bucket_name)
            print(f"Connected to Bucket: {bucket_name} ✅")
        return cls._bucket

# Short-hand helpers for imports
def get_cluster(): return CouchbaseManager.get_cluster()
def get_bucket(): return CouchbaseManager.get_bucket()
