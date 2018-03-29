CREATE OR REPLACE FUNCTION retrieve_entryid_exact(hw varchar) RETURNS SETOF idxlexalpdeu.value%TYPE AS $$ 
DECLARE
  contrib idxlexalpdeu%ROWTYPE;
  status varchar;
  eid varchar;
BEGIN 
  FOR contrib IN SELECT * FROM idxlexalpdeu  
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = hw AND idxlexalpdeu.lang = 'deu' LOOP 
    SELECT into status idxlexalpdeu.value from idxlexalpdeu where entryid=contrib.entryid and key = 'cdm-contribution-status';
    IF status = 'modified' OR status='finished' THEN
      SELECT into eid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=contrib.entryid and idxlexalpdeu.key = 'cdm-entry-id';
      RETURN NEXT eid;
    END IF;
  END LOOP; 

  
  RETURN; 
END; 
$$ LANGUAGE plpgsql;





CREATE OR REPLACE FUNCTION retrieve_idxdeu_exact(hw varchar) RETURNS SETOF idxlexalpdeu AS $$ 
DECLARE
  contrib idxlexalpdeu%ROWTYPE;
  status varchar;
  eid varchar;
BEGIN 
  FOR contrib IN SELECT * FROM idxlexalpdeu  
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = hw AND idxlexalpdeu.lang = 'deu' LOOP 
    SELECT into status idxlexalpdeu.value from idxlexalpdeu where entryid=contrib.entryid and key = 'cdm-contribution-status';
    IF status = 'modified' OR status='finished' THEN
      -- SELECT into eid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=contrib.entryid and idxlexalpdeu.key = 'cdm-entry-id';
      RETURN NEXT contrib;
    END IF;
  END LOOP; 

  
  RETURN; 
END; 
$$ LANGUAGE plpgsql;




CREATE OR REPLACE FUNCTION retrieve_entryid_fuzzy(hw varchar) RETURNS SETOF idxlexalpdeu.value%TYPE AS $$ 
DECLARE
  contrib idxlexalpdeu%ROWTYPE;
  status varchar;
  eid varchar;
BEGIN 
  FOR contrib IN SELECT * FROM idxlexalpdeu  
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value like hw AND idxlexalpdeu.lang = 'deu' LOOP 
    SELECT into status idxlexalpdeu.value from idxlexalpdeu where entryid=contrib.entryid and key = 'cdm-contribution-status';
    IF status = 'modified' OR status='finished' THEN
      SELECT into eid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=contrib.entryid and idxlexalpdeu.key = 'cdm-entry-id';
      RETURN NEXT eid;
    END IF;
  END LOOP; 

  
  RETURN; 
END; 
$$ LANGUAGE plpgsql;



CREATE OR REPLACE FUNCTION retrieve_idxdeu_fuzzy(hw varchar) RETURNS SETOF idxlexalpdeu AS $$ 
DECLARE
  contrib idxlexalpdeu%ROWTYPE;
  status varchar;
  eid varchar;
BEGIN 
  FOR contrib IN SELECT * FROM idxlexalpdeu  
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value like hw AND idxlexalpdeu.lang = 'deu' LOOP 
    SELECT into status idxlexalpdeu.value from idxlexalpdeu where entryid=contrib.entryid and key = 'cdm-contribution-status';
    IF status = 'modified' OR status='finished' THEN
      -- SELECT into eid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=contrib.entryid and idxlexalpdeu.key = 'cdm-entry-id';
      RETURN NEXT contrib;
    END IF;
  END LOOP; 
  
  RETURN; 
END; 
$$ LANGUAGE plpgsql;




CREATE OR REPLACE FUNCTION isCurrent_deu(eid varchar) RETURNS boolean AS $F$
DECLARE
  status varchar;
BEGIN
  SELECT into status idxlexalpdeu.value from idxlexalpdeu where entryid=eid and key = 'cdm-contribution-status';
  RETURN (status = 'modified' OR status='finished');
END;
$F$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION get_entry_id_deu(eid varchar) RETURNS idxlexalpdeu.value%TYPE AS $F$
DECLARE
  cdmeid varchar;
BEGIN
  SELECT into cdmeid idxlexalpdeu.value FROM idxlexalpdeu where idxlexalpdeu.entryid=eid and idxlexalpdeu.key = 'cdm-entry-id';
  RETURN cdmeid;
END;
$F$ LANGUAGE plpgsql;





CREATE OR REPLACE FUNCTION add_accessor_functions() RETURNS integer AS $PROC$
DECLARE
  nbv integer := 0;
  dic varchar;
  dbn varchar; 
  lg varchar;
BEGIN 
  FOR dic in SELECT name from dictionaries WHERE type='pivot' LOOP 
    FOR dbn, lg IN SELECT dbname, sourcelanguage FROM volumes LOOP
      IF lg != 'axi' THEN
        PERFORM add_accessor_functions('idx' || dbn, lg);
      END IF;
    END LOOP;
  END LOOP;
  RETURN nbv;
END;
$PROC$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION add_accessor_functions(idxdbn varchar, lang varchar) RETURNS integer AS $PROC$
DECLARE
  ftxt varchar;
BEGIN 
  ftxt :=         $$ CREATE OR REPLACE FUNCTION retrieve_entryid_exact_$$ || lang || $$(hw varchar) RETURNS SETOF $$ || idxdbn || $$.value%TYPE AS $F$ $$;
  ftxt := ftxt || $$ DECLARE $$;
  ftxt := ftxt || $$  contrib $$ || idxdbn || $$%ROWTYPE; $$;
  ftxt := ftxt || $$  status varchar; $$;
  ftxt := ftxt || $$  eid varchar; $$;
  ftxt := ftxt || $$ BEGIN $$;
  
  ftxt := ftxt || $$   FOR contrib IN SELECT * FROM $$ || idxdbn ;
  ftxt := ftxt || $$       WHERE key = 'cdm-headword' AND value = hw AND lang = '$$ || lang || $$' LOOP $$;
  ftxt := ftxt || $$     SELECT into status $$ || idxdbn || $$.value from $$ || idxdbn || $$ WHERE entryid=contrib.entryid and key = 'cdm-contribution-status'; $$;  
  ftxt := ftxt || $$     IF status = 'modified' OR status='finished' THEN $$;  
  ftxt := ftxt || $$     SELECT into eid $$ || idxdbn || $$.value FROM $$ || idxdbn || $$ where $$ || idxdbn || $$.entryid=contrib.entryid and $$ || idxdbn || $$.key = 'cdm-entry-id'; $$;  
  ftxt := ftxt || $$     RETURN NEXT eid; $$;  

  ftxt := ftxt || $$     END IF; $$;  

  ftxt := ftxt || $$   END LOOP; $$;  
  ftxt := ftxt || $$   RETURN; $$;
  ftxt := ftxt || $$ END; $$;
  ftxt := ftxt || $$ $F$ LANGUAGE plpgsql; $$;

  EXECUTE ftxt;
  RETURN 0;
END;
$PROC$ LANGUAGE plpgsql;


-- NOTE: see if plperl is possible to maintain a hash of already queried axies and avoid re querying axie's status.

CREATE OR REPLACE FUNCTION retrieve_axie_linked_to(hw varchar, lang varchar) RETURNS SETOF idxlexalpaxi.entryid%TYPE AS $P$
DECLARE
  contrib idxlexalpaxi.entryid%TYPE;
  status varchar;
  eid varchar;
BEGIN
  FOR contrib IN EXECUTE $$SELECT distinct entryid FROM idxlexalpaxi  $$ ||
       $$ WHERE key = 'axi-reflexie' AND value IN (select * from retrieve_entryid_exact_$$ || lang || $$('$$ || hw || $$'))$$ LOOP 
    SELECT INTO status value FROM idxlexalpaxi WHERE idxlexalpaxi.entryid=contrib AND key = 'cdm-contribution-status';
    IF status = 'modified' OR status='finished' THEN
      RETURN NEXT contrib;
    END IF;
  END LOOP; 

  
  RETURN; 
END;
$P$ LANGUAGE plpgsql;





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




