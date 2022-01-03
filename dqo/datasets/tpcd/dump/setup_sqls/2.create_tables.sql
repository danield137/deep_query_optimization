create table if not exists dss_part
(
    p_partkey     int primary key not null,
    p_name        varchar(55)     null,
    p_mfgr        char(25)        null,
    p_brand       char(10)        null,
    p_type        varchar(25)     null,
    p_size        int             null,
    p_container   char(10)        null,
    p_retailprice decimal(19, 4)  null,
    p_comment     varchar(23)     null
);

create table if not exists dss_region
(
    r_regionkey int primary key not null,
    r_name      char(25)        null,
    r_comment   varchar(152)    null
);

create table if not exists dss_nation
(
    n_nationkey int primary key not null,
    n_name      char(25)        null,
    n_regionkey int             null,
    n_comment   varchar(152)    null
);

create table if not exists dss_customer
(
    c_custkey    bigint primary key not null,
    c_name       varchar(25)        null,
    c_address    varchar(40)        null,
    c_nationkey  int                null,
    c_phone      char(15)           null,
    c_acctbal    decimal(19, 4)     null,
    c_mktsegment char(10)           null,
    c_comment    varchar(118)       null
);

create table if not exists dss_order
(
    o_orderkey      decimal primary key not null,
    o_custkey       int                 not null,
    o_orderstatus   char(1)             null,
    o_totalprice    double precision    null,
    o_orderdate     date                null,
    o_orderpriority char(15)            null,
    o_clerk         char(15)            null,
    o_shippriority  int                 null,
    o_comment       varchar(79)         null
);

create table if not exists dss_supplier
(
    s_suppkey   int primary key not null,
    s_name      char(25)        null,
    s_address   varchar(40)     null,
    s_nationkey int             null,
    s_phone     char(15)        null,
    s_acctbal   decimal(19, 4)  null,
    s_comment   varchar(102)    null
);

create table if not exists dss_partsupp
(
    ps_partkey    int              not null,
    ps_suppkey    int              not null,
    ps_availqty   int              null,
    ps_supplycost double precision not null,
    ps_comment    varchar(199)     null,
    primary key (ps_partkey, ps_suppkey)
);

create table if not exists dss_lineitem
(
    l_orderkey      decimal          not null,
    l_partkey       int              not null,
    l_suppkey       int              not null,
    l_linenumber    int              not null,
    l_quantity      double precision not null,
    l_extendedprice double precision not null,
    l_discount      double precision not null,
    l_tax           double precision not null,
    l_returnflag    char(1)          null,
    l_linestatus    char(1)          null,
    l_shipdate      date             null,
    l_commitdate    date             null,
    l_receiptdate   date             null,
    l_shipinstruct  char(25)         null,
    l_shipmode      char(10)         null,
    l_comment       varchar(44)      null,
    primary key (l_orderkey, l_linenumber)
);
