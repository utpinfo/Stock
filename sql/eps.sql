drop table eps;

create table eps (
    eps_id int auto_increment primary key,
    stock_code varchar(50) not null,
	eps_date date not null,
	eps float not null,
    entry_id varchar(100),
    entry_date timestamp default current_timestamp,
    tr_id varchar(100),
    tr_date timestamp default current_timestamp
);


alter table eps
add constraint eps_uk unique (stock_code,eps_date);
