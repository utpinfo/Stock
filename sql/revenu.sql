drop table revenue;

create table revenue (
    revenue_id int auto_increment primary key,
    stock_code varchar(50) not null,
	revenue_date date not null,
	revenue bigint not null,
    entry_id varchar(100),
    entry_date timestamp default current_timestamp,
    tr_id varchar(100),
    tr_date timestamp default current_timestamp
);


alter table revenue
add constraint revenue_uk unique (stock_code,revenue_date);
