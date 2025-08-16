#!/bin/bash

set +e
pkill -9 -f etcd
pkill -9 -f python
rm -rf default.etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://10.220.80.13:2379 &
