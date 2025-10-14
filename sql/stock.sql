drop table stock;

create table stock (
    stock_id int auto_increment primary key,
    stock_code varchar(50) not null,
    stock_name varchar(100) not null,
	stock_kind varchar(30) not null,
	isin_code varchar(60) not null,
	stock_status varchar(2) default '00',
    entry_id varchar(100),
    entry_date timestamp default current_timestamp,
    tr_id varchar(100),
    tr_date timestamp default current_timestamp
);


alter table stock
add constraint stock_uk unique (stock_code);
