select lexalpdeu.xmlcode from lexalpdeu WHERE 
  lexalpdeu.ObjectId IN 
  ( select distinct idxlexalpdeu.entryId from idxlexalpdeu 
    WHERE  
    ( idxlexalpdeu.entryId IN 
      ( select idxlexalpdeu.entryId from idxlexalpdeu 
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value ilike 'Landschaft' AND idxlexalpdeu.lang = 'deu' 
      )
    )  AND  
    ( idxlexalpdeu.entryId IN 
      ( select idxlexalpdeu.entryId from idxlexalpdeu 
        WHERE idxlexalpdeu.key = 'cdm-contribution-status' AND idxlexalpdeu.value = 'finished' 
      )  OR 
      idxlexalpdeu.entryId IN 
      ( select idxlexalpdeu.entryId from idxlexalpdeu 
        WHERE idxlexalpdeu.key = 'cdm-contribution-status' AND idxlexalpdeu.value = 'modified'
      )  
    )  
  )  
  ORDER BY multilingual_sort('fra',headword)  LIMIT 10 OFFSET 0


select distinct idxlexalpaxi.entryid from idxlexalpaxi where 
  idxlexalpaxi.value in (
		select distinct idxlexalpdeu.value from idxlexalpdeu where
			idxlexalpdeu.key = 'cdm-entry-id' AND
			idxlexalpdeu.entryId in
			( select distinct idxlexalpdeu.entryId from idxlexalpdeu 
				WHERE  
				
				( idxlexalpdeu.entryId IN 
					( select idxlexalpdeu.entryId from idxlexalpdeu 
						WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value ilike 'Landschaft' AND idxlexalpdeu.lang = 'deu' 
					)
				)  AND  
				( idxlexalpdeu.entryId IN 
					( select idxlexalpdeu.entryId from idxlexalpdeu 
            WHERE idxlexalpdeu.key = 'cdm-contribution-status' AND idxlexalpdeu.value = 'finished' 
          )  OR 
          idxlexalpdeu.entryId IN 
          ( select idxlexalpdeu.entryId from idxlexalpdeu 
            WHERE idxlexalpdeu.key = 'cdm-contribution-status' AND idxlexalpdeu.value = 'modified'
          )  
        )  
      ) and
      idxlexalpaxi.key='axi-reflexie' and 
    ( idxlexalpaxi.entryId IN 
      ( select idxlexalpaxi.entryId from idxlexalpaxi 
        WHERE idxlexalpaxi.key = 'cdm-contribution-status' AND idxlexalpaxi.value = 'finished' 
      )  OR 
      idxlexalpaxi.entryId IN 
      ( select idxlexalpaxi.entryId from idxlexalpaxi 
        WHERE idxlexalpaxi.key = 'cdm-contribution-status' AND idxlexalpaxi.value = 'modified'
      )  
    )
  )
  
CREATE OR REPLACE FUNCTION retrieve_entryid_exact(hw varchar) RETURNS SETOF idxlexalpdeu.value%TYPE AS $$ 
DECLARE
  contrib idxlexalpdeu%ROWTYPE;
  status varchar;
  eid varchar;
BEGIN 
  FOR contrib IN SELECT * FROM idxlexalpdeu  
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = hw AND idxlexalpdeu.lang = 'deu' LOOP 
    select into status idxlexalpdeu.value from idxlexalpdeu where entryid=contrib.entryid and key = 'cdm-contribution-status';
    IF status = 'modified' OR status='finished' THEN
      SELECT into eid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=contrib.entryid and idxlexalpdeu.key = 'cdm-entry-id';
      RETURN NEXT eid;
    END IF;
  END LOOP; 

  
  RETURN; 
END; 
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION retrieve_entryid_exact(hw varchar) RETURNS SETOF varchar AS $$ 
DECLARE
  contrib idxlexalpdeu%ROWTYPE;
  status varchar;
  entryid varchar;
BEGIN 
  FOR contrib IN SELECT * FROM idxlexalpdeu  
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = hw AND idxlexalpdeu.lang = 'deu' LOOP 
    select into status idxlexalpdeu.value from idxlexalpdeu where entryid=contrib.entryid and key = 'cdm-contribution-status';
    RAISE NOTICE 'trying % -> status %', contrib.entryid, status;    
    IF status = 'modified' OR status='finished' THEN
      RAISE NOTICE 'found a match';
      select into entryid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=contrib.entryid and idxlexalpdeu.key = 'cdm-entry-id';
      RETURN NEXT entryid;
    END IF;
  END LOOP; 

  
  RETURN; 
END; 
$$ LANGUAGE plpgsql;


select * from retrieve_axie_exact('Landschaft');

select lexalpdeu.xmlcode from lexalpdeu WHERE 
  lexalpdeu.ObjectId IN (
  select * from retrieve_axie_exact('Landschaft') ) 
  ORDER BY multilingual_sort('fra',headword)  LIMIT 10 OFFSET 0;
  
select distinct idxlexalpaxi.entryid from idxlexalpaxi where 
  idxlexalpaxi.value in (
    select distinct idxlexalpdeu.value from idxlexalpdeu where
      idxlexalpdeu.key = 'cdm-entry-id' AND
      idxlexalpdeu.entryId in 
      ( select * from retrieve_axie_exact('Landschaft') ) and
      idxlexalpaxi.key='axi-reflexie' and 
    ( idxlexalpaxi.entryId IN 
      ( select idxlexalpaxi.entryId from idxlexalpaxi 
        WHERE idxlexalpaxi.key = 'cdm-contribution-status' AND idxlexalpaxi.value = 'finished' 
      )  OR 
      idxlexalpaxi.entryId IN 
      ( select idxlexalpaxi.entryId from idxlexalpaxi 
        WHERE idxlexalpaxi.key = 'cdm-contribution-status' AND idxlexalpaxi.value = 'modified'
      )  
    )
  )
  
  
select lexalpdeu.objectid from lexalpdeu WHERE 
  lexalpdeu.ObjectId IN (
  select * from retrieve_axie_exact('Landschaft') ) 