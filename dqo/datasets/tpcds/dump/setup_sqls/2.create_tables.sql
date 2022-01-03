create table customer_address
(
	ca_address_sk int not null
		primary key,
	ca_address_id char(16) not null,
	ca_street_number char(10) null,
	ca_street_name varchar(60) null,
	ca_street_type char(15) null,
	ca_suite_number char(10) null,
	ca_city varchar(60) null,
	ca_county varchar(30) null,
	ca_state char(2) null,
	ca_zip char(10) null,
	ca_country varchar(20) null,
	ca_gmt_offset decimal(5,2) null,
	ca_location_type char(20) null
);

create table customer_demographics
(
	cd_demo_sk int not null
		primary key,
	cd_gender char null,
	cd_marital_status char null,
	cd_education_status char(20) null,
	cd_purchase_estimate int null,
	cd_credit_rating char(10) null,
	cd_dep_count int null,
	cd_dep_employed_count int null,
	cd_dep_college_count int null
);

create table date_dim
(
	d_date_sk int not null
		primary key,
	d_date_id char(16) not null,
	d_date date null,
	d_month_seq int null,
	d_week_seq int null,
	d_quarter_seq int null,
	d_year int null,
	d_dow int null,
	d_moy int null,
	d_dom int null,
	d_qoy int null,
	d_fy_year int null,
	d_fy_quarter_seq int null,
	d_fy_week_seq int null,
	d_day_name char(9) null,
	d_quarter_name char(6) null,
	d_holiday char null,
	d_weekend char null,
	d_following_holiday char null,
	d_first_dom int null,
	d_last_dom int null,
	d_same_day_ly int null,
	d_same_day_lq int null,
	d_current_day char null,
	d_current_week char null,
	d_current_month char null,
	d_current_quarter char null,
	d_current_year char null
);

create table call_center
(
	cc_call_center_sk int not null primary key,
	cc_call_center_id char(16) not null,
	cc_rec_start_date date null,
	cc_rec_end_date date null,
	cc_closed_date_sk int null,
	cc_open_date_sk int null,
	cc_name varchar(50) null,
	cc_class varchar(50) null,
	cc_employees int null,
	cc_sq_ft int null,
	cc_hours char(20) null,
	cc_manager varchar(40) null,
	cc_mkt_id int null,
	cc_mkt_class char(50) null,
	cc_mkt_desc varchar(100) null,
	cc_market_manager varchar(40) null,
	cc_division int null,
	cc_division_name varchar(50) null,
	cc_company int null,
	cc_company_name char(50) null,
	cc_street_number char(10) null,
	cc_street_name varchar(60) null,
	cc_street_type char(15) null,
	cc_suite_number char(10) null,
	cc_city varchar(60) null,
	cc_county varchar(30) null,
	cc_state char(2) null,
	cc_zip char(10) null,
	cc_country varchar(20) null,
	cc_gmt_offset decimal(5,2) null,
	cc_tax_percentage decimal(5,2) null
);

create table catalog_page
(
	cp_catalog_page_sk int not null
		primary key,
	cp_catalog_page_id char(16) not null,
	cp_start_date_sk int null,
	cp_end_date_sk int null,
	cp_department varchar(50) null,
	cp_catalog_number int null,
	cp_catalog_page_number int null,
	cp_description varchar(100) null,
	cp_type varchar(100) null
);

create table income_band
(
	ib_income_band_sk int not null
		primary key,
	ib_lower_bound int null,
	ib_upper_bound int null
);

create table household_demographics
(
	hd_demo_sk int not null
		primary key,
	hd_income_band_sk int null,
	hd_buy_potential char(15) null,
	hd_dep_count int null,
	hd_vehicle_count int null
);

create table customer
(
	c_customer_sk int not null
		primary key,
	c_customer_id char(16) not null,
	c_current_cdemo_sk int null,
	c_current_hdemo_sk int null,
	c_current_addr_sk int null,
	c_first_shipto_date_sk int null,
	c_first_sales_date_sk int null,
	c_salutation char(10) null,
	c_first_name char(20) null,
	c_last_name char(30) null,
	c_preferred_cust_flag char null,
	c_birth_day int null,
	c_birth_month int null,
	c_birth_year int null,
	c_birth_country varchar(20) null,
	c_login char(13) null,
	c_email_address char(50) null,
	c_last_review_date char(10) null
);

create table item
(
	i_item_sk int not null
		primary key,
	i_item_id char(16) not null,
	i_rec_start_date date null,
	i_rec_end_date date null,
	i_item_desc varchar(200) null,
	i_current_price decimal(7,2) null,
	i_wholesale_cost decimal(7,2) null,
	i_brand_id int null,
	i_brand char(50) null,
	i_class_id int null,
	i_class char(50) null,
	i_category_id int null,
	i_category char(50) null,
	i_manufact_id int null,
	i_manufact char(50) null,
	i_size char(20) null,
	i_formulation char(20) null,
	i_color char(20) null,
	i_units char(10) null,
	i_container char(10) null,
	i_manager_id int null,
	i_product_name char(50) null
);

create table promotion
(
	p_promo_sk int not null
		primary key,
	p_promo_id char(16) not null,
	p_start_date_sk int null,
	p_end_date_sk int null,
	p_item_sk int null,
	p_cost decimal(15,2) null,
	p_response_target int null,
	p_promo_name char(50) null,
	p_channel_dmail char null,
	p_channel_email char null,
	p_channel_catalog char null,
	p_channel_tv char null,
	p_channel_radio char null,
	p_channel_press char null,
	p_channel_event char null,
	p_channel_demo char null,
	p_channel_details varchar(100) null,
	p_purpose char(15) null,
	p_discount_active char null
);

create table reason
(
	r_reason_sk int not null
		primary key,
	r_reason_id char(16) not null,
	r_reason_desc char(100) null
);

create table ship_mode
(
	sm_ship_mode_sk int not null
		primary key,
	sm_ship_mode_id char(16) not null,
	sm_type char(30) null,
	sm_code char(10) null,
	sm_carrier char(20) null,
	sm_contract char(20) null
);

create table store
(
	s_store_sk int not null
		primary key,
	s_store_id char(16) not null,
	s_rec_start_date date null,
	s_rec_end_date date null,
	s_closed_date_sk int null,
	s_store_name varchar(50) null,
	s_number_employees int null,
	s_floor_space int null,
	s_hours char(20) null,
	s_manager varchar(40) null,
	s_market_id int null,
	s_geography_class varchar(100) null,
	s_market_desc varchar(100) null,
	s_market_manager varchar(40) null,
	s_division_id int null,
	s_division_name varchar(50) null,
	s_company_id int null,
	s_company_name varchar(50) null,
	s_street_number varchar(10) null,
	s_street_name varchar(60) null,
	s_street_type char(15) null,
	s_suite_number char(10) null,
	s_city varchar(60) null,
	s_county varchar(30) null,
	s_state char(2) null,
	s_zip char(10) null,
	s_country varchar(20) null,
	s_gmt_offset decimal(5,2) null,
	s_tax_precentage decimal(5,2) null
);

create table time_dim
(
	t_time_sk int not null
		primary key,
	t_time_id char(16) not null,
	t_time int null,
	t_hour int null,
	t_minute int null,
	t_second int null,
	t_am_pm char(2) null,
	t_shift char(20) null,
	t_sub_shift char(20) null,
	t_meal_time char(20) null
);

create table store_returns
(
	sr_returned_date_sk int null,
	sr_return_time_sk int null,
	sr_item_sk int not null,
	sr_customer_sk int null,
	sr_cdemo_sk int null,
	sr_hdemo_sk int null,
	sr_addr_sk int null,
	sr_store_sk int null,
	sr_reason_sk int null,
	sr_ticket_number int not null,
	sr_return_quantity int null,
	sr_return_amt decimal(7,2) null,
	sr_return_tax decimal(7,2) null,
	sr_return_amt_inc_tax decimal(7,2) null,
	sr_fee decimal(7,2) null,
	sr_return_ship_cost decimal(7,2) null,
	sr_refunded_cash decimal(7,2) null,
	sr_reversed_charge decimal(7,2) null,
	sr_store_credit decimal(7,2) null,
	sr_net_loss decimal(7,2) null,
	primary key (sr_item_sk, sr_ticket_number)
);

create table store_sales
(
	ss_sold_date_sk int null,
	ss_sold_time_sk int null,
	ss_item_sk int not null,
	ss_customer_sk int null,
	ss_cdemo_sk int null,
	ss_hdemo_sk int null,
	ss_addr_sk int null,
	ss_store_sk int null,
	ss_promo_sk int null,
	ss_ticket_number int not null,
	ss_quantity int null,
	ss_wholesale_cost decimal(7,2) null,
	ss_list_price decimal(7,2) null,
	ss_sales_price decimal(7,2) null,
	ss_ext_discount_amt decimal(7,2) null,
	ss_ext_sales_price decimal(7,2) null,
	ss_ext_wholesale_cost decimal(7,2) null,
	ss_ext_list_price decimal(7,2) null,
	ss_ext_tax decimal(7,2) null,
	ss_coupon_amt decimal(7,2) null,
	ss_net_paid decimal(7,2) null,
	ss_net_paid_inc_tax decimal(7,2) null,
	ss_net_profit decimal(7,2) null,
	primary key (ss_item_sk, ss_ticket_number)
);

create table warehouse
(
	w_warehouse_sk int not null
		primary key,
	w_warehouse_id char(16) not null,
	w_warehouse_name varchar(20) null,
	w_warehouse_sq_ft int null,
	w_street_number char(10) null,
	w_street_name varchar(60) null,
	w_street_type char(15) null,
	w_suite_number char(10) null,
	w_city varchar(60) null,
	w_county varchar(30) null,
	w_state char(2) null,
	w_zip char(10) null,
	w_country varchar(20) null,
	w_gmt_offset decimal(5,2) null
);

create table catalog_returns
(
	cr_returned_date_sk int null,
	cr_returned_time_sk int null,
	cr_item_sk int not null,
	cr_refunded_customer_sk int null,
	cr_refunded_cdemo_sk int null,
	cr_refunded_hdemo_sk int null,
	cr_refunded_addr_sk int null,
	cr_returning_customer_sk int null,
	cr_returning_cdemo_sk int null,
	cr_returning_hdemo_sk int null,
	cr_returning_addr_sk int null,
	cr_call_center_sk int null,
	cr_catalog_page_sk int null,
	cr_ship_mode_sk int null,
	cr_warehouse_sk int null,
	cr_reason_sk int null,
	cr_order_number int not null,
	cr_return_quantity int null,
	cr_return_amount decimal(7,2) null,
	cr_return_tax decimal(7,2) null,
	cr_return_amt_inc_tax decimal(7,2) null,
	cr_fee decimal(7,2) null,
	cr_return_ship_cost decimal(7,2) null,
	cr_refunded_cash decimal(7,2) null,
	cr_reversed_charge decimal(7,2) null,
	cr_store_credit decimal(7,2) null,
	cr_net_loss decimal(7,2) null,
	primary key (cr_item_sk, cr_order_number)
);

create table catalog_sales
(
	cs_sold_date_sk int null,
	cs_sold_time_sk int null,
	cs_ship_date_sk int null,
	cs_bill_customer_sk int null,
	cs_bill_cdemo_sk int null,
	cs_bill_hdemo_sk int null,
	cs_bill_addr_sk int null,
	cs_ship_customer_sk int null,
	cs_ship_cdemo_sk int null,
	cs_ship_hdemo_sk int null,
	cs_ship_addr_sk int null,
	cs_call_center_sk int null,
	cs_catalog_page_sk int null,
	cs_ship_mode_sk int null,
	cs_warehouse_sk int null,
	cs_item_sk int not null,
	cs_promo_sk int null,
	cs_order_number int not null,
	cs_quantity int null,
	cs_wholesale_cost decimal(7,2) null,
	cs_list_price decimal(7,2) null,
	cs_sales_price decimal(7,2) null,
	cs_ext_discount_amt decimal(7,2) null,
	cs_ext_sales_price decimal(7,2) null,
	cs_ext_wholesale_cost decimal(7,2) null,
	cs_ext_list_price decimal(7,2) null,
	cs_ext_tax decimal(7,2) null,
	cs_coupon_amt decimal(7,2) null,
	cs_ext_ship_cost decimal(7,2) null,
	cs_net_paid decimal(7,2) null,
	cs_net_paid_inc_tax decimal(7,2) null,
	cs_net_paid_inc_ship decimal(7,2) null,
	cs_net_paid_inc_ship_tax decimal(7,2) null,
	cs_net_profit decimal(7,2) null,
	primary key (cs_item_sk, cs_order_number)
);

create table inventory
(
	inv_date_sk int not null,
	inv_item_sk int not null,
	inv_warehouse_sk int not null,
	inv_quantity_on_hand int null,
	primary key (inv_date_sk, inv_item_sk, inv_warehouse_sk)

);

create table web_page
(
	wp_web_page_sk int not null
		primary key,
	wp_web_page_id char(16) not null,
	wp_rec_start_date date null,
	wp_rec_end_date date null,
	wp_creation_date_sk int null,
	wp_access_date_sk int null,
	wp_autogen_flag char null,
	wp_customer_sk int null,
	wp_url varchar(100) null,
	wp_type char(50) null,
	wp_char_count int null,
	wp_link_count int null,
	wp_image_count int null,
	wp_max_ad_count int null
);

create table web_returns
(
	wr_returned_date_sk int null,
	wr_returned_time_sk int null,
	wr_item_sk int not null,
	wr_refunded_customer_sk int null,
	wr_refunded_cdemo_sk int null,
	wr_refunded_hdemo_sk int null,
	wr_refunded_addr_sk int null,
	wr_returning_customer_sk int null,
	wr_returning_cdemo_sk int null,
	wr_returning_hdemo_sk int null,
	wr_returning_addr_sk int null,
	wr_web_page_sk int null,
	wr_reason_sk int null,
	wr_order_number int not null,
	wr_return_quantity int null,
	wr_return_amt decimal(7,2) null,
	wr_return_tax decimal(7,2) null,
	wr_return_amt_inc_tax decimal(7,2) null,
	wr_fee decimal(7,2) null,
	wr_return_ship_cost decimal(7,2) null,
	wr_refunded_cash decimal(7,2) null,
	wr_reversed_charge decimal(7,2) null,
	wr_account_credit decimal(7,2) null,
	wr_net_loss decimal(7,2) null,
	primary key (wr_item_sk, wr_order_number)
);

create table web_site
(
	web_site_sk int not null
		primary key,
	web_site_id char(16) not null,
	web_rec_start_date date null,
	web_rec_end_date date null,
	web_name varchar(50) null,
	web_open_date_sk int null,
	web_close_date_sk int null,
	web_class varchar(50) null,
	web_manager varchar(40) null,
	web_mkt_id int null,
	web_mkt_class varchar(50) null,
	web_mkt_desc varchar(100) null,
	web_market_manager varchar(40) null,
	web_company_id int null,
	web_company_name char(50) null,
	web_street_number char(10) null,
	web_street_name varchar(60) null,
	web_street_type char(15) null,
	web_suite_number char(10) null,
	web_city varchar(60) null,
	web_county varchar(30) null,
	web_state char(2) null,
	web_zip char(10) null,
	web_country varchar(20) null,
	web_gmt_offset decimal(5,2) null,
	web_tax_percentage decimal(5,2) null
);

create table web_sales
(
	ws_sold_date_sk int null,
	ws_sold_time_sk int null,
	ws_ship_date_sk int null,
	ws_item_sk int not null,
	ws_bill_customer_sk int null,
	ws_bill_cdemo_sk int null,
	ws_bill_hdemo_sk int null,
	ws_bill_addr_sk int null,
	ws_ship_customer_sk int null,
	ws_ship_cdemo_sk int null,
	ws_ship_hdemo_sk int null,
	ws_ship_addr_sk int null,
	ws_web_page_sk int null,
	ws_web_site_sk int null,
	ws_ship_mode_sk int null,
	ws_warehouse_sk int null,
	ws_promo_sk int null,
	ws_order_number int not null,
	ws_quantity int null,
	ws_wholesale_cost decimal(7,2) null,
	ws_list_price decimal(7,2) null,
	ws_sales_price decimal(7,2) null,
	ws_ext_discount_amt decimal(7,2) null,
	ws_ext_sales_price decimal(7,2) null,
	ws_ext_wholesale_cost decimal(7,2) null,
	ws_ext_list_price decimal(7,2) null,
	ws_ext_tax decimal(7,2) null,
	ws_coupon_amt decimal(7,2) null,
	ws_ext_ship_cost decimal(7,2) null,
	ws_net_paid decimal(7,2) null,
	ws_net_paid_inc_tax decimal(7,2) null,
	ws_net_paid_inc_ship decimal(7,2) null,
	ws_net_paid_inc_ship_tax decimal(7,2) null,
	ws_net_profit decimal(7,2) null,
	primary key (ws_item_sk, ws_order_number)
);

