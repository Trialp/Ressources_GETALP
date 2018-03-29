CREATE OR REPLACE FUNCTION get_distance_select_nf(d integer, tbl text, lang text, status text) 
RETURNS text AS $$ 
DECLARE 
tt text;
BEGIN 

tt := 'INSERT INTO ' || tbl || ' SELECT entryid, ' || d  || ' from idxlexalp' || lang || ' where value in (select distinct pcnf.value from idxlexalp' || lang || ' as f ' ||
       ' inner join idxlexalp' || lang || ' as pcnf on pcnf.entryid=f.entryid ' ||
       ' where pcnf.key=''cdm-previous-classified-' || status || '-contribution'' and ';
IF d = 1 THEN
  tt := tt || ' f.key=''cdm-contribution-status'' and (f.value=''finished'' or f.value=''modified''))';
ELSE
  tt := tt || ' f.entryid in (select contribid from ' || tbl || ' where dist=' || (d-1) || '))';
END IF;
tt := tt || ' and key=''cdm-contribution-id'';'; 
RETURN tt;
END; 
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION populate_distances(tbl text, lang text, status text) 
RETURNS text AS $$ 
DECLARE 
oldnbe integer := 0;
nbe integer := 1;
distance integer := 1;
updated boolean := true;
tt text = '';
BEGIN
  EXECUTE 'CREATE TABLE ' || tbl || ' (contribid character varying(255), dist integer)';
  WHILE nbe > oldnbe LOOP
    -- RAISE NOTICE 'updating : %' , get_distance_select_nf(distance, tbl, lang, status);
    EXECUTE get_distance_select_nf(distance, tbl, lang, status);
    -- RAISE NOTICE 'found: %', FOUND;
    oldnbe := nbe;
    EXECUTE 'SELECT count(contribid) from ' || tbl INTO nbe;
    distance := distance + 1;
    -- RAISE NOTICE 'nbe: %', nbe;
  END LOOP;
  RAISE NOTICE 'Populated % with % entries', tbl, nbe;
RETURN tt;
END; 
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION populate_all_distances()
RETURNS text AS $$ 
DECLARE
BEGIN
PERFORM populate_distances('d_nf_fra', 'fra', 'not-finished');
PERFORM populate_distances('d_nf_slv', 'slv', 'not-finished');
PERFORM populate_distances('d_nf_ita', 'ita', 'not-finished');
PERFORM populate_distances('d_nf_deu', 'deu', 'not-finished');
PERFORM populate_distances('d_nf_axi', 'axi', 'not-finished');
PERFORM populate_distances('d_nf_eng', 'eng', 'not-finished');

PERFORM populate_distances('d_f_fra', 'fra', 'finished');
PERFORM populate_distances('d_f_slv', 'slv', 'finished');
PERFORM populate_distances('d_f_ita', 'ita', 'finished');
PERFORM populate_distances('d_f_deu', 'deu', 'finished');
PERFORM populate_distances('d_f_axi', 'axi', 'finished');
PERFORM populate_distances('d_f_eng', 'eng', 'finished');

RETURN '';
END;
$$ LANGUAGE plpgsql;

-- select populate_all_distances();

CREATE OR REPLACE FUNCTION purge_lexalp(tbl text, lang text, status text)
RETURNS text AS $$
DECLARE
nbbefore integer := 0;
nbafter  integer := 0;
tt text;
BEGIN
  tt := 'delete from lexalp' || lang || ' where objectid in ' ||
        ' (select entryid from idxlexalp' || lang  || ' where key=''cdm-contribution-status'' and value=''classified ' || status || 
        ''' and entryid not in (select contribid from '|| tbl || ' where dist < 3))';
  EXECUTE 'SELECT count(objectid) from lexalp' || lang INTO nbbefore;
  EXECUTE tt;
  EXECUTE 'SELECT count(objectid) from lexalp' || lang INTO nbafter;
  RAISE NOTICE 'Deleted: % entries.', nbbefore - nbafter;
  tt := 'delete from idxlexalp' || lang || ' where entryid not in (select objectid from lexalp'  || lang || ')';
  EXECUTE 'SELECT count(objectid) from idxlexalp' || lang INTO nbbefore;
  EXECUTE tt;
  EXECUTE 'SELECT count(objectid) from idxlexalp' || lang INTO nbafter;
  RAISE NOTICE 'Deleted: % indexes.', nbbefore - nbafter;
  RETURN '';
END;
$$ LANGUAGE plpgsql;

-- select purge_lexalp('d_nf_fra', 'fra', 'not finished');


CREATE OR REPLACE FUNCTION purge_all()
RETURNS text AS $$ 
DECLARE
BEGIN
PERFORM purge_lexalp('d_nf_fra', 'fra', 'not finished');
PERFORM purge_lexalp('d_nf_slv', 'slv', 'not finished');
PERFORM purge_lexalp('d_nf_ita', 'ita', 'not finished');
PERFORM purge_lexalp('d_nf_deu', 'deu', 'not finished');
PERFORM purge_lexalp('d_nf_eng', 'eng', 'not finished');

PERFORM purge_lexalp('d_f_fra', 'fra', 'finished');
PERFORM purge_lexalp('d_f_slv', 'slv', 'finished');
PERFORM purge_lexalp('d_f_ita', 'ita', 'finished');
PERFORM purge_lexalp('d_f_deu', 'deu', 'finished');
PERFORM purge_lexalp('d_f_eng', 'eng', 'finished');

PERFORM purge_lexalp('d_f_axi', 'axi', 'finished');
PERFORM purge_lexalp('d_nf_axi', 'axi', 'not finished');

RETURN '';
END;
$$ LANGUAGE plpgsql;

-- select purge_all();

