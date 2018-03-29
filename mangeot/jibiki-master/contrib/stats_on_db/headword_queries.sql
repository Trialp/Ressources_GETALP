CREATE VIEW hw_lexalpfra
AS SELECT DISTINCT i.value FROM idxlexalpfra AS i
     INNER JOIN idxlexalpfra AS status ON status.entryid=i.entryid
   WHERE status.key='cdm-contribution-status' AND (status.value='finished' OR status.value='modified') AND i.key='cdm-headword';

CREATE VIEW hw_lexalpdeu
AS SELECT DISTINCT i.value FROM idxlexalpdeu AS i
     INNER JOIN idxlexalpdeu AS status ON status.entryid=i.entryid
   WHERE status.key='cdm-contribution-status' AND (status.value='finished' OR status.value='modified') AND i.key='cdm-headword';

CREATE VIEW hw_lexalpita
AS SELECT DISTINCT i.value FROM idxlexalpita AS i
     INNER JOIN idxlexalpita AS status ON status.entryid=i.entryid
   WHERE status.key='cdm-contribution-status' AND (status.value='finished' OR status.value='modified') AND i.key='cdm-headword';

CREATE VIEW hw_lexalpslv
AS SELECT DISTINCT i.value FROM idxlexalpslv AS i
     INNER JOIN idxlexalpslv AS status ON status.entryid=i.entryid
   WHERE status.key='cdm-contribution-status' AND (status.value='finished' OR status.value='modified') AND i.key='cdm-headword';

CREATE VIEW hw_lexalpeng
AS SELECT DISTINCT i.value FROM idxlexalpeng AS i
     INNER JOIN idxlexalpeng AS status ON status.entryid=i.entryid
   WHERE status.key='cdm-contribution-status' AND (status.value='finished' OR status.value='modified') AND i.key='cdm-headword';

CREATE INDEX idxlexalpfra_value_idx ON idxlexalpfra (value);
CREATE INDEX idxlexalpdeu_value_idx ON idxlexalpdeu (value);
CREATE INDEX idxlexalpita_value_idx ON idxlexalpita (value);
CREATE INDEX idxlexalpslv_value_idx ON idxlexalpslv (value);
CREATE INDEX idxlexalpeng_value_idx ON idxlexalpeng (value);

ANALYZE idxlexalpfra;
ANALYZE idxlexalpdeu;
ANALYZE idxlexalpita;
ANALYZE idxlexalpslv;
ANALYZE idxlexalpeng;


