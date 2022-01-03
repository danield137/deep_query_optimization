#!/bin/bash

while getopts ":h:u:" opt; do
  case $opt in
    h) host="$OPTARG"
    ;;
    u) user="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

psql -h ${host} -U ${user} -W --set=sslmode=require -d postgres -f ./setup_sqls/1.create_db.sql
echo 'step 1: tpch created'
psql -h ${host} -U ${user} -W --set=sslmode=require -d tpch -f ./setup_sqls/2.create_tables.sql
echo 'step 2: tables created'
psql -h ${host} -U ${user} --set=sslmode=require -W -d tpch -f ./setup_sqls/3.import_data.sql
echo 'step 3: data imported'
psql -h ${host} -U ${user} --set=sslmode=require -W -d tpch -f ./setup_sqls/4.create_indexes.sql
echo 'step 4: indexed'
psql -h ${host} -U ${user} --set=sslmode=require -W -d tpch -f ./setup_sqls/5.extras.sql
echo 'step 5: extras'
