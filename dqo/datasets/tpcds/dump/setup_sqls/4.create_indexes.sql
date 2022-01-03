create index cc_d1
	on call_center (cc_closed_date_sk);

create index cc_d2
	on call_center (cc_open_date_sk);

create index cp_d1
	on catalog_page (cp_end_date_sk);

create index cp_d2
	on catalog_page (cp_start_date_sk);

create index cr_a1
	on catalog_returns (cr_refunded_addr_sk);

create index cr_a2
	on catalog_returns (cr_returning_addr_sk);

create index cr_c1
	on catalog_returns (cr_refunded_customer_sk);

create index cr_c2
	on catalog_returns (cr_returning_customer_sk);

create index cr_cc
	on catalog_returns (cr_call_center_sk);

create index cr_cd1
	on catalog_returns (cr_refunded_cdemo_sk);

create index cr_cd2
	on catalog_returns (cr_returning_cdemo_sk);

create index cr_cp
	on catalog_returns (cr_catalog_page_sk);

create index cr_d1
	on catalog_returns (cr_returned_date_sk);

create index cr_hd1
	on catalog_returns (cr_refunded_hdemo_sk);

create index cr_hd2
	on catalog_returns (cr_returning_hdemo_sk);

create index cr_r
	on catalog_returns (cr_reason_sk);

create index cr_sm
	on catalog_returns (cr_ship_mode_sk);

create index cr_t
	on catalog_returns (cr_returned_time_sk);

create index cr_w2
	on catalog_returns (cr_warehouse_sk);

create index cs_b_a
	on catalog_sales (cs_bill_addr_sk);

create index cs_b_c
	on catalog_sales (cs_bill_customer_sk);

create index cs_b_cd
	on catalog_sales (cs_bill_cdemo_sk);

create index cs_b_hd
	on catalog_sales (cs_bill_hdemo_sk);

create index cs_cc
	on catalog_sales (cs_call_center_sk);

create index cs_cp
	on catalog_sales (cs_catalog_page_sk);

create index cs_d1
	on catalog_sales (cs_ship_date_sk);

create index cs_d2
	on catalog_sales (cs_sold_date_sk);

create index cs_p
	on catalog_sales (cs_promo_sk);

create index cs_s_a
	on catalog_sales (cs_ship_addr_sk);

create index cs_s_c
	on catalog_sales (cs_ship_customer_sk);

create index cs_s_cd
	on catalog_sales (cs_ship_cdemo_sk);

create index cs_s_hd
	on catalog_sales (cs_ship_hdemo_sk);

create index cs_sm
	on catalog_sales (cs_ship_mode_sk);

create index cs_t
	on catalog_sales (cs_sold_time_sk);

create index cs_w
	on catalog_sales (cs_warehouse_sk);

create index c_a
	on customer (c_current_addr_sk);

create index c_cd
	on customer (c_current_cdemo_sk);

create index c_fsd
	on customer (c_first_sales_date_sk);

create index c_fsd2
	on customer (c_first_shipto_date_sk);

create index c_hd
	on customer (c_current_hdemo_sk);

create index hd_ib
	on household_demographics (hd_income_band_sk);

create index inv_i
	on inventory (inv_item_sk);

create index inv_w
	on inventory (inv_warehouse_sk);

create index p_end_date
	on promotion (p_end_date_sk);

create index p_i
	on promotion (p_item_sk);

create index p_start_date
	on promotion (p_start_date_sk);

create index s_close_date
	on store (s_closed_date_sk);

create index sr_a
	on store_returns (sr_addr_sk);

create index sr_c
	on store_returns (sr_customer_sk);

create index sr_cd
	on store_returns (sr_cdemo_sk);

create index sr_hd
	on store_returns (sr_hdemo_sk);

create index sr_r
	on store_returns (sr_reason_sk);

create index sr_ret_d
	on store_returns (sr_returned_date_sk);

create index sr_s
	on store_returns (sr_store_sk);

create index sr_t
	on store_returns (sr_return_time_sk);

create index ss_a
	on store_sales (ss_addr_sk);

create index ss_c
	on store_sales (ss_customer_sk);

create index ss_cd
	on store_sales (ss_cdemo_sk);

create index ss_d
	on store_sales (ss_sold_date_sk);

create index ss_hd
	on store_sales (ss_hdemo_sk);

create index ss_p
	on store_sales (ss_promo_sk);

create index ss_s
	on store_sales (ss_store_sk);

create index ss_t
	on store_sales (ss_sold_time_sk);

create index wp_ad
	on web_page (wp_access_date_sk);

create index wp_cd
	on web_page (wp_creation_date_sk);

create index wr_r
	on web_returns (wr_reason_sk);

create index wr_ref_a
	on web_returns (wr_refunded_addr_sk);

create index wr_ref_c
	on web_returns (wr_refunded_customer_sk);

create index wr_ref_cd
	on web_returns (wr_refunded_cdemo_sk);

create index wr_ref_hd
	on web_returns (wr_refunded_hdemo_sk);

create index wr_ret_a
	on web_returns (wr_returning_addr_sk);

create index wr_ret_c
	on web_returns (wr_returning_customer_sk);

create index wr_ret_cd
	on web_returns (wr_returning_cdemo_sk);

create index wr_ret_d
	on web_returns (wr_returned_date_sk);

create index wr_ret_hd
	on web_returns (wr_returning_hdemo_sk);

create index wr_ret_t
	on web_returns (wr_returned_time_sk);

create index wr_wp
	on web_returns (wr_web_page_sk);

create index ws_b_a
	on web_sales (ws_bill_addr_sk);

create index ws_b_c
	on web_sales (ws_bill_customer_sk);

create index ws_b_cd
	on web_sales (ws_bill_cdemo_sk);

create index ws_b_hd
	on web_sales (ws_bill_hdemo_sk);

create index ws_d2
	on web_sales (ws_sold_date_sk);

create index ws_p
	on web_sales (ws_promo_sk);

create index ws_s_a
	on web_sales (ws_ship_addr_sk);

create index ws_s_c
	on web_sales (ws_ship_customer_sk);

create index ws_s_cd
	on web_sales (ws_ship_cdemo_sk);

create index ws_s_d
	on web_sales (ws_ship_date_sk);

create index ws_s_hd
	on web_sales (ws_ship_hdemo_sk);

create index ws_sm
	on web_sales (ws_ship_mode_sk);

create index ws_t
	on web_sales (ws_sold_time_sk);

create index ws_w2
	on web_sales (ws_warehouse_sk);

create index ws_wp
	on web_sales (ws_web_page_sk);

create index ws_ws
	on web_sales (ws_web_site_sk);

create index web_d1
	on web_site (web_close_date_sk);

create index web_d2
	on web_site (web_open_date_sk);









