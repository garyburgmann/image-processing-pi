import redis

from app.config import REDIS

pool = redis.ConnectionPool(host=REDIS.host, port=REDIS.port, db=REDIS.db)


def get_redis() -> redis.Redis:
    """ attach to a different redis db for cache """
    global pool
    return redis.Redis(connection_pool=pool, decode_responses=True)
