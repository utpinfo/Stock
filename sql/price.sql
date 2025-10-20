drop table price;

create table price (
    price_id int auto_increment primary key,
    stock_code varchar(50) not null,
    price_date date,
	open float not null,
	close float not null,
	high float not null,
	low float not null,
	volume bigint,
    entry_id varchar(100),
    entry_date timestamp default current_timestamp,
    tr_id varchar(100),
    tr_date timestamp default current_timestamp
);


alter table price
add constraint price_uk unique (stock_code,price_date);
