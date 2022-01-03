create index if not exists c_nationkey on dss_customer (c_nationkey);
create index if not exists n_regionkey on dss_nation (n_regionkey);
create index if not exists o_custkey on dss_order (o_custkey);
create index if not exists l_partkey on dss_lineitem (l_partkey, l_suppkey);
create index if not exists l_suppkey on dss_lineitem (l_suppkey, l_partkey);
create index if not exists ps_suppkey on dss_partsupp (ps_suppkey);

