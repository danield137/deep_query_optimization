create table if not exists part
(
    p_partkey     bigint         not null,
    p_type        varchar(25)    null,
    p_size        int            null,
    p_brand       char(10)       null,
    p_name        varchar(55)    null,
    p_container   char(10)       null,
    p_mfgr        char(25)       null,
    p_retailprice decimal(19, 4) null,
    p_comment     varchar(23)    null,
    primary key (p_partkey)
);

create table if not exists region
(
    r_regionkey int          not null,
    r_name      char(25)     null,
    r_comment   varchar(152) null,
    primary key (r_regionkey)
);

create table if not exists nation
(
    n_nationkey int          not null,
    n_name      char(25)     null,
    n_regionkey int          null,
    n_comment   varchar(152) null,
    primary key (n_nationkey),
    constraint nation_ibfk_1
        foreign key (n_regionkey) references region (r_regionkey)
            on update cascade on delete cascade
);

create table if not exists customer
(
    c_custkey    bigint         not null,
    c_mktsegment char(10)       null,
    c_nationkey  int            null,
    c_name       varchar(25)    null,
    c_address    varchar(40)    null,
    c_phone      char(15)       null,
    c_acctbal    decimal(19, 4) null,
    c_comment    varchar(118)   null,
    primary key (c_custkey),
    constraint customer_ibfk_1
        foreign key (c_nationkey) references nation (n_nationkey)
            on update cascade on delete cascade
);

create index if not exists c_nationkey
    on customer (c_nationkey);

create index if not exists n_regionkey
    on nation (n_regionkey);

create table if not exists orders
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
    primary key (o_orderkey),
    constraint orders_ibfk_1
        foreign key (o_custkey) references customer (c_custkey)
            on update cascade on delete cascade
);

create index if not exists o_custkey
    on orders (o_custkey);

create table if not exists supplier
(
    s_suppkey   int            not null,
    s_nationkey int            null,
    s_comment   varchar(102)   null,
    s_name      char(25)       null,
    s_address   varchar(40)    null,
    s_phone     char(15)       null,
    s_acctbal   decimal(19, 4) null,
    primary key (s_suppkey),
    constraint supplier_ibfk_1
        foreign key (s_nationkey) references nation (n_nationkey)
);

create table if not exists partsupp
(
    ps_partkey    bigint         not null,
    ps_suppkey    int            not null,
    ps_supplycost decimal(19, 4) not null,
    ps_availqty   int            null,
    ps_comment    varchar(199)   null,
    primary key (ps_partkey, ps_suppkey),
    constraint partsupp_ibfk_1
        foreign key (ps_partkey) references part (p_partkey)
            on update cascade on delete cascade,
    constraint partsupp_ibfk_2
        foreign key (ps_suppkey) references supplier (s_suppkey)
            on update cascade on delete cascade
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
    primary key (l_orderkey, l_linenumber),
    constraint lineitem_ibfk_1
        foreign key (l_orderkey) references orders (o_orderkey)
            on update cascade on delete cascade,
    constraint lineitem_ibfk_2
        foreign key (l_partkey, l_suppkey) references partsupp (ps_partkey, ps_suppkey)
            on update cascade on delete cascade
);

create index if not exists l_partkey
    on lineitem (l_partkey, l_suppkey);

create index if not exists l_suppkey
    on lineitem (l_suppkey, l_partkey);

create index if not exists ps_suppkey
    on partsupp (ps_suppkey);

