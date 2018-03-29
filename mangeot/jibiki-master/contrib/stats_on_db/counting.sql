select count(value) from idxlexalpaxi where key='cdm-entry-id' and (status='finished' or status='modified');

select value from idxlexalpaxi 
where 
  	key='cdm-axie-reflexie'
  and
  	entryid in (select entryid, from idxlexalpaxi where key='cdm-entry-id' and (status='finished' or status='modified'));
  and 
  	(status='finished' or status='modified');
  	
  	
select i2.value, i1.value from idxlexalpaxi as i1 left join idxlexalpaxi as i2 on i1.entryid=i2.entryid where i1.key='axi-reflexie' and i2.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');
  	 --> 11356 rows
  	 

select i2.value, i1.value, f.entryid from idxlexalpaxi as i1 
   left join idxlexalpaxi as i2 on i1.entryid=i2.entryid 
   left join idxlexalpfra as f on i2.value=f.value
where i1.key='axi-reflexie' and i1.lang='fra' and i2.key='cdm-entry-id' and f.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');

select f.entryid, i1.entryid from idxlexalpfra as f where f.key='cdm-entry-id' and f.value in 
 ( select value from idxlexalpaxi as i1 where i1.key='axi-reflexie' and i1.lang='fra' and (i1.status='finished' or i1.status='modified')
 )
and (f.status='finished' or f.status='modified');

select i2.value, i1.value, f.entryid from idxlexalpaxi as i1 
   left join idxlexalpaxi as i2 on i1.entryid=i2.entryid 
   inner join idxlexalpfra as f on i2.value=f.value
where i1.key='axi-reflexie' and i1.lang='fra' and i2.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');


----


create view alfra as select i2.value as axieid, i1.value as lexieid from idxlexalpaxi i1 
   left join idxlexalpaxi  i2 on i1.entryid=i2.entryid 
where i1.key='axi-reflexie' and i1.lang='fra' and i2.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');

create view aldeu as select i2.value as axieid, i1.value as lexieid from idxlexalpaxi i1 
   inner join idxlexalpaxi  i2 on i1.entryid=i2.entryid 
where i1.key='axi-reflexie' and i1.lang='deu' and i2.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');

create view alita as select i2.value as axieid, i1.value as lexieid from idxlexalpaxi i1 
   inner join idxlexalpaxi  i2 on i1.entryid=i2.entryid 
where i1.key='axi-reflexie' and i1.lang='ita' and i2.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');

create view alslv as select i2.value as axieid, i1.value as lexieid from idxlexalpaxi i1 
   inner join idxlexalpaxi  i2 on i1.entryid=i2.entryid 
where i1.key='axi-reflexie' and i1.lang='slv' and i2.key='cdm-entry-id' and (i2.status='finished' or i2.status='modified');

create view fradom as select f1.value as lexieid, f2.value as domain , f3.value as harmo, f1.entryid as entryid
  from idxlexalpfra f1 
       inner join idxlexalpfra f2 on f1.entryid = f2.entryid
       inner join idxlexalpfra f3 on f1.entryid = f3.entryid
  where f1.key='cdm-entry-id' and f2.key ='lexalp-domain' and f3.key='lexalp-harmonising-status'  and (f1.status='finished' or f1.status='modified');

create view deudom as select f1.value as lexieid, f2.value as domain , f3.value as harmo, f1.entryid as entryid
  from idxlexalpdeu f1 
       inner join idxlexalpdeu f2 on f1.entryid = f2.entryid
       inner join idxlexalpdeu f3 on f1.entryid = f3.entryid
  where f1.key='cdm-entry-id' and f2.key ='lexalp-domain' and f3.key='lexalp-harmonising-status'  and (f1.status='finished' or f1.status='modified');

create view itadom as select f1.value as lexieid, f2.value as domain , f3.value as harmo, f1.entryid as entryid
  from idxlexalpita f1 
       inner join idxlexalpita f2 on f1.entryid = f2.entryid
       inner join idxlexalpita f3 on f1.entryid = f3.entryid
  where f1.key='cdm-entry-id' and f2.key ='lexalp-domain' and f3.key='lexalp-harmonising-status'  and (f1.status='finished' or f1.status='modified');

create view slvdom as select f1.value as lexieid, f2.value as domain , f3.value as harmo, f1.entryid as entryid
  from idxlexalpslv f1 
       inner join idxlexalpslv f2 on f1.entryid = f2.entryid
       inner join idxlexalpslv f3 on f1.entryid = f3.entryid
  where f1.key='cdm-entry-id' and f2.key ='lexalp-domain' and f3.key='lexalp-harmonising-status'  and (f1.status='finished' or f1.status='modified');


select al.axieid, al.lexieid, f.entryid from alfra al inner join idxlexalpfra f on al.lexieid = f.value where f.key='cdm-entry-id' and (f.status='finished' or f.status='modified');