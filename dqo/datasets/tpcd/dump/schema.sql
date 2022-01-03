create table if not exists dss_part
(
    p_partkey     int            not null,
    p_name        varchar(55)    null,
    p_mfgr        char(25)       null,
    p_brand       char(10)       null,
    p_type        varchar(25)    null,
    p_size        int            null,
    p_container   char(10)       null,
    p_retailprice decimal(19, 4) null,
    p_comment     varchar(23)    null,
    primary key (p_partkey)
);

create table if not exists dss_region
(
    r_regionkey int          not null,
    r_name      char(25)     null,
    r_comment   varchar(152) null,
    primary key (r_regionkey)
);

create table if not exists dss_nation
(
    n_nationkey int          not null,
    n_name      char(25)     null,
    n_regionkey int          null,
    n_comment   varchar(152) null,
    primary key (n_nationkey)
);

create table if not exists dss_customer
(
    c_custkey    int            not null,
    c_name       varchar(25)    null,
    c_address    varchar(40)    null,
    c_nationkey  int            null,
    c_phone      char(15)       null,
    c_acctbal    decimal(19, 4) null,
    c_mktsegment char(10)       null,
    c_comment    varchar(117)   null,
    primary key (c_custkey)
);

create index if not exists c_nationkey
    on dss_customer (c_nationkey);

create index if not exists n_regionkey
    on dss_nation (n_regionkey);

create table if not exists dss_order
(
    o_orderdate     date           null,
    o_orderkey      bigint         not null,
    o_custkey       bigint         not null,
    o_orderpriority char(15)       null,
    o_shippriority  int            null,
    o_clerk         char(15)       null,
    o_orderstatus   char           null,
    o_totalprice    decimal(19, 4) null,
    o_comment       varchar(79)    null,
    primary key (o_orderkey)
);

create index if not exists o_custkey
    on dss_order (o_custkey);

create table if not exists dss_supplier
(
    s_suppkey   int            not null,
    s_nationkey int            null,
    s_comment   varchar(102)   null,
    s_name      char(25)       null,
    s_address   varchar(40)    null,
    s_phone     char(15)       null,
    s_acctbal   decimal(19, 4) null,
    primary key (s_suppkey)
);

create table if not exists dss_partsupp
(
    ps_partkey    bigint         not null,
    ps_suppkey    int            not null,
    ps_supplycost decimal(19, 4) not null,
    ps_availqty   int            null,
    ps_comment    varchar(199)   null,
    primary key (ps_partkey, ps_suppkey)
);

create table if not exists dss_lineitem
(
    l_shipdate      date           null,
    l_orderkey      bigint         not null,
    l_discount      decimal(19, 4) not null,
    l_extendedprice decimal(19, 4) not null,
    l_suppkey       int            not null,
    l_quantity      bigint         not null,
    l_returnflag    char           null,
    l_partkey       bigint         not null,
    l_linestatus    char           null,
    l_tax           decimal(19, 4) not null,
    l_commitdate    date           null,
    l_receiptdate   date           null,
    l_shipmode      char(10)       null,
    l_linenumber    bigint         not null,
    l_shipinstruct  char(25)       null,
    l_comment       varchar(44)    null,
    primary key (l_orderkey, l_linenumber)
);

create index if not exists l_partkey
    on dss_lineitem (l_partkey, l_suppkey);

create index if not exists l_suppkey
    on dss_lineitem (l_suppkey, l_partkey);

create index if not exists ps_suppkey
    on dss_partsupp (ps_suppkey);

