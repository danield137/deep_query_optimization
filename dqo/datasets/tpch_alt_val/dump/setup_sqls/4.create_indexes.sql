create index if not exists c_nationkey on customer (c_nationkey);
create index if not exists n_regionkey on nation (n_regionkey);
create index if not exists o_custkey on orders (o_custkey);
create index if not exists l_partkey on lineitem (l_partkey, l_suppkey);
create index if not exists l_suppkey on lineitem (l_suppkey, l_partkey);
create index if not exists ps_suppkey on partsupp (ps_suppkey);

