
explain analyze select idxlexalpdeu.value from idxlexalpdeu WHERE 
  idxlexalpdeu.key='cdm-entry-id' AND
  idxlexalpdeu.entryId IN 
  ( select distinct idxlexalpdeu.entryId from idxlexalpdeu 
    WHERE  
    ( idxlexalpdeu.entryId IN 
      ( select idxlexalpdeu.entryId from idxlexalpdeu 
        WHERE idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = 'Landschaft' AND idxlexalpdeu.lang = 'deu' 
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
  );
  
  
--- RETRIEVE idx entries

explain analyze select * from idxlexalpdeu 
  where idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = 'Landschaft' AND isCurrent_deu(idxlexalpdeu.entryid);

explain analyze select * from retrieve_idxdeu_exact('Landschaft');

explain analyze select * from idxlexalpdeu 
  where idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value like '%Landschaft%' AND isCurrent_deu(idxlexalpdeu.entryid);

explain analyze select * from retrieve_idxdeu_fuzzy('%Landschaft%');

--- RETRIEVE entryids

explain analyze select get_entry_id_deu(idxlexalpdeu.entryid) from idxlexalpdeu 
  where idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value = 'Landschaft' AND isCurrent_deu(idxlexalpdeu.entryid);

explain analyze select * from retrieve_entryid_exact('Landschaft');

explain analyze select get_entry_id_deu(entryid) from idxlexalpdeu 
  where idxlexalpdeu.key = 'cdm-headword' AND idxlexalpdeu.value like '%Landschaft%' 
  AND isCurrent_deu(idxlexalpdeu.entryid);

explain analyze select * from retrieve_entryid_fuzzy('%Landschaft%');
