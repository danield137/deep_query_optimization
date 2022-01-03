create table if not exists part
(
    p_partkey     bigint primary key not null,
    p_type        varchar(25)        null,
    p_size        int                null,
    p_brand       char(10)           null,
    p_name        varchar(55)        null,
    p_container   char(10)           null,
    p_mfgr        char(25)           null,
    p_retailprice decimal(19, 4)     null,
    p_comment     varchar(23)        null
);

create table if not exists region
(
    r_regionkey int primary key not null,
    r_name      char(25)        null,
    r_comment   varchar(152)    null
);

create table if not exists nation
(
    n_nationkey int primary key not null,
    n_name      char(25)        null,
    n_regionkey int             null,
    n_comment   varchar(152)    null
);

create table if not exists customer
(
    c_custkey    bigint primary key not null,
    c_mktsegment char(10)           null,
    c_nationkey  int                null,
    c_name       varchar(25)        null,
    c_address    varchar(40)        null,
    c_phone      char(15)           null,
    c_acctbal    decimal(19, 4)     null,
    c_comment    varchar(118)       null
);

create table if not exists orders
(
    o_orderdate     date               null,
    o_orderkey      bigint primary key not null,
    o_custkey       bigint             not null,
    o_orderpriority char(15)           null,
    o_shippriority  int                null,
    o_clerk         char(15)           null,
    o_orderstatus   char               null,
    o_totalprice    decimal(19, 4)     null,
    o_comment       varchar(79)        null
);

create table if not exists supplier
(
    s_suppkey   int primary key not null,
    s_nationkey int             null,
    s_comment   varchar(102)    null,
    s_name      char(25)        null,
    s_address   varchar(40)     null,
    s_phone     char(15)        null,
    s_acctbal   decimal(19, 4)  null
);

create table if not exists partsupp
(
    ps_partkey    bigint         not null,
    ps_suppkey    int            not null,
    ps_supplycost decimal(19, 4) not null,
    ps_availqty   int            null,
    ps_comment    varchar(199)   null,
    primary key (ps_partkey, ps_suppkey)
);

create table if not exists lineitem
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
