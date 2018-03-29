-- $Id$
-- in order to load this function, execute: createlang plpgsql papillon
-- then use as is:
-- select headword from gdefest order by est_sort(headword);
-- or create an index:
--
CREATE OR REPLACE LANGUAGE plpgsql;


-- bam Bambara
-- a b c d e ɛ f g ɲ h i j k l m n ŋ o ɔ p q r s t u v w x y z
CREATE OR REPLACE FUNCTION bam_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
 i		integer;
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'C' THEN
    result:= result || '03';
  ELSIF tmp = 'c' THEN
    result:= result || '03';
  ELSIF tmp = 'D' THEN
    result:= result || '04';
  ELSIF tmp = 'd' THEN
    result:= result || '04';
  ELSIF tmp = 'E' THEN
    result:= result || '05';
  ELSIF tmp = 'e' THEN
    result:= result || '05';
  ELSIF tmp = 'Ɛ' THEN
    result:= result || '06';
  ELSIF tmp = 'ɛ' THEN
    result:= result || '06';
  ELSIF tmp = 'F' THEN
    result:= result || '07';
  ELSIF tmp = 'f' THEN
    result:= result || '07';
  ELSIF tmp = 'G' THEN
    result:= result || '08';
  ELSIF tmp = 'g' THEN
    result:= result || '08';
  ELSIF tmp = 'Ɲ' THEN
    result:= result || '09';
  ELSIF tmp = 'ɲ' THEN
    result:= result || '09';
  ELSIF tmp = 'H' THEN
    result:= result || '10';
  ELSIF tmp = 'h' THEN
    result:= result || '10';
  ELSIF tmp = 'I' THEN
    result:= result || '11';
  ELSIF tmp = 'i' THEN
    result:= result || '11';
  ELSIF tmp = 'J' THEN
    result:= result || '12';
  ELSIF tmp = 'j' THEN
    result:= result || '12';
  ELSIF tmp = 'K' THEN
    result:= result || '13';
  ELSIF tmp = 'k' THEN
    result:= result || '13';
  ELSIF tmp = 'L' THEN
    result:= result || '14';
  ELSIF tmp = 'l' THEN
    result:= result || '14';
  ELSIF tmp = 'M' THEN
    result:= result || '15';
  ELSIF tmp = 'm' THEN
    result:= result || '15';
  ELSIF tmp = 'N' THEN
    result:= result || '16';
  ELSIF tmp = 'n' THEN
    result:= result || '16';
  ELSIF tmp = 'Ŋ' THEN
    result:= result || '17';
  ELSIF tmp = 'ŋ' THEN
    result:= result || '17';
  ELSIF tmp = 'O' THEN
    result:= result || '18';
  ELSIF tmp = 'o' THEN
    result:= result || '18';
  ELSIF tmp = 'Ɔ' THEN
    result:= result || '19';
  ELSIF tmp = 'ɔ' THEN
    result:= result || '19';
  ELSIF tmp = 'p' THEN
    result:= result || '20';
  ELSIF tmp = 'P' THEN
    result:= result || '20';
  ELSIF tmp = 'Q' THEN
    result:= result || '21';
  ELSIF tmp = 'q' THEN
    result:= result || '21';
  ELSIF tmp = 'R' THEN
    result:= result || '22';
  ELSIF tmp = 'r' THEN
    result:= result || '22';
  ELSIF tmp = 'S' THEN
    result:= result || '23';
  ELSIF tmp = 's' THEN
    result:= result || '23';
  ELSIF tmp = 'T' THEN
    result:= result || '24';
  ELSIF tmp = 't' THEN
    result:= result || '24';
  ELSIF tmp = 'U' THEN
    result:= result || '25';
  ELSIF tmp = 'u' THEN
    result:= result || '25';
  ELSIF tmp = 'V' THEN
    result:= result || '26';
  ELSIF tmp = 'v' THEN
    result:= result || '26';
  ELSIF tmp = 'W' THEN
    result:= result || '27';
  ELSIF tmp = 'w' THEN
    result:= result || '27';
  ELSIF tmp = 'X' THEN
    result:= result || '28';
  ELSIF tmp = 'x' THEN
    result:= result || '28';
  ELSIF tmp = 'Y' THEN
    result:= result || '29';
  ELSIF tmp = 'y' THEN
    result:= result || '29';
  ELSIF tmp = 'Z' THEN
    result:= result || '30';
  ELSIF tmp = 'z' THEN
    result:= result || '30';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );



-- deu German
-- (a ä) b c d e f g h i j k l m n (o ö) p q r (s ß) t (u ü) v w x y z
-- for the moment do nothing
-- todo: umlaut and other diacritics ...

CREATE OR REPLACE FUNCTION deu_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
 i		integer;
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'À' THEN
    result:= result || '01';
  ELSIF tmp = 'à' THEN
    result:= result || '01';
  ELSIF tmp = 'Â' THEN
    result:= result || '01';
  ELSIF tmp = 'â' THEN
    result:= result || '01';
  ELSIF tmp = 'Ä' THEN
    result:= result || '01';
  ELSIF tmp = 'ä' THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'C' THEN
    result:= result || '03';
  ELSIF tmp = 'c' THEN
    result:= result || '03';
  ELSIF tmp = 'Ç' THEN
    result:= result || '03';
  ELSIF tmp = 'ç' THEN
    result:= result || '03';
  ELSIF tmp = 'D' THEN
    result:= result || '04';
  ELSIF tmp = 'd' THEN
    result:= result || '04';
  ELSIF tmp = 'E' THEN
    result:= result || '05';
  ELSIF tmp = 'e' THEN
    result:= result || '05';
  ELSIF tmp = 'É' THEN
    result:= result || '05';
  ELSIF tmp = 'é' THEN
    result:= result || '05';
  ELSIF tmp = 'È' THEN
    result:= result || '05';
  ELSIF tmp = 'è' THEN
    result:= result || '05';
  ELSIF tmp = 'Ê' THEN
    result:= result || '05';
  ELSIF tmp = 'ê' THEN
    result:= result || '05';
  ELSIF tmp = 'Ë' THEN
    result:= result || '05';
  ELSIF tmp = 'ë' THEN
    result:= result || '05';
  ELSIF tmp = 'F' THEN
    result:= result || '06';
  ELSIF tmp = 'f' THEN
    result:= result || '06';
  ELSIF tmp = 'G' THEN
    result:= result || '07';
  ELSIF tmp = 'g' THEN
    result:= result || '07';
  ELSIF tmp = 'H' THEN
    result:= result || '08';
  ELSIF tmp = 'h' THEN
    result:= result || '08';
  ELSIF tmp = 'I' THEN
    result:= result || '09';
  ELSIF tmp = 'i' THEN
    result:= result || '09';
  ELSIF tmp = 'Î' THEN
    result:= result || '09';
  ELSIF tmp = 'î' THEN
    result:= result || '09';
  ELSIF tmp = 'Ï' THEN
    result:= result || '09';
  ELSIF tmp = 'ï' THEN
    result:= result || '09';
  ELSIF tmp = 'J' THEN
    result:= result || '10';
  ELSIF tmp = 'j' THEN
    result:= result || '10';
  ELSIF tmp = 'K' THEN
    result:= result || '11';
  ELSIF tmp = 'k' THEN
    result:= result || '11';
  ELSIF tmp = 'L' THEN
    result:= result || '12';
  ELSIF tmp = 'l' THEN
    result:= result || '12';
  ELSIF tmp = 'M' THEN
    result:= result || '13';
  ELSIF tmp = 'm' THEN
    result:= result || '13';
  ELSIF tmp = 'N' THEN
    result:= result || '14';
  ELSIF tmp = 'n' THEN
    result:= result || '14';
  ELSIF tmp = 'O' THEN
    result:= result || '15';
  ELSIF tmp = 'o' THEN
    result:= result || '15';
  ELSIF tmp = 'Ö' THEN
    result:= result || '15';
  ELSIF tmp = 'ö' THEN
    result:= result || '15';
  ELSIF tmp = 'ô' THEN
    result:= result || '15';
  ELSIF tmp = 'Ô' THEN
    result:= result || '15';
  ELSIF tmp = 'p' THEN
    result:= result || '16';
  ELSIF tmp = 'P' THEN
    result:= result || '16';
  ELSIF tmp = 'Q' THEN
    result:= result || '17';
  ELSIF tmp = 'q' THEN
    result:= result || '17';
  ELSIF tmp = 'R' THEN
    result:= result || '18';
  ELSIF tmp = 'r' THEN
    result:= result || '18';
  ELSIF tmp = 'S' THEN
    result:= result || '19';
  ELSIF tmp = 's' THEN
    result:= result || '19';
  ELSIF tmp = 'ß' THEN
    result:= result || '1919';
  ELSIF tmp = 'T' THEN
    result:= result || '20';
  ELSIF tmp = 't' THEN
    result:= result || '20';
  ELSIF tmp = 'U' THEN
    result:= result || '22';
  ELSIF tmp = 'u' THEN
    result:= result || '22';
  ELSIF tmp = 'Ù' THEN
    result:= result || '22';
  ELSIF tmp = 'ù' THEN
    result:= result || '22';
  ELSIF tmp = 'Û' THEN
    result:= result || '22';
  ELSIF tmp = 'û' THEN
    result:= result || '22';
  ELSIF tmp = 'Ü' THEN
    result:= result || '22';
  ELSIF tmp = 'ü' THEN
    result:= result || '22';
  ELSIF tmp = 'V' THEN
    result:= result || '23';
  ELSIF tmp = 'v' THEN
    result:= result || '23';
  ELSIF tmp = 'W' THEN
    result:= result || '24';
  ELSIF tmp = 'w' THEN
    result:= result || '24';
  ELSIF tmp = 'X' THEN
    result:= result || '25';
  ELSIF tmp = 'x' THEN
    result:= result || '25';
  ELSIF tmp = 'Y' THEN
    result:= result || '26';
  ELSIF tmp = 'y' THEN
    result:= result || '26';
  ELSIF tmp = 'Z' THEN
    result:= result || '27';
  ELSIF tmp = 'z' THEN
    result:= result || '27';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );

-- dje Soŋay-zarma
-- a, b, c, d, e, f, g, h, i, j, k, l, m, n, Ɲ ɲ, ŋ, o, p, r, s, Š š, t, u, w, y, z, Ž ž

CREATE OR REPLACE FUNCTION dje_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
 i		integer;
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'C' THEN
    result:= result || '03';
  ELSIF tmp = 'c' THEN
    result:= result || '03';
  ELSIF tmp = 'D' THEN
    result:= result || '04';
  ELSIF tmp = 'd' THEN
    result:= result || '04';
  ELSIF tmp = 'E' THEN
    result:= result || '05';
  ELSIF tmp = 'e' THEN
    result:= result || '05';
  ELSIF tmp = 'F' THEN
    result:= result || '06';
  ELSIF tmp = 'f' THEN
    result:= result || '06';
  ELSIF tmp = 'G' THEN
    result:= result || '07';
  ELSIF tmp = 'g' THEN
    result:= result || '07';
  ELSIF tmp = 'H' THEN
    result:= result || '08';
  ELSIF tmp = 'h' THEN
    result:= result || '08';
  ELSIF tmp = 'I' THEN
    result:= result || '09';
  ELSIF tmp = 'i' THEN
    result:= result || '09';
  ELSIF tmp = 'J' THEN
    result:= result || '10';
  ELSIF tmp = 'j' THEN
    result:= result || '10';
  ELSIF tmp = 'K' THEN
    result:= result || '11';
  ELSIF tmp = 'k' THEN
    result:= result || '11';
  ELSIF tmp = 'L' THEN
    result:= result || '12';
  ELSIF tmp = 'l' THEN
    result:= result || '12';
  ELSIF tmp = 'M' THEN
    result:= result || '13';
  ELSIF tmp = 'm' THEN
    result:= result || '13';
  ELSIF tmp = 'N' THEN
    result:= result || '14';
  ELSIF tmp = 'n' THEN
    result:= result || '14';
  ELSIF tmp = 'Ɲ' THEN
    result:= result || '15';
  ELSIF tmp = 'ɲ' THEN
    result:= result || '15';
  ELSIF tmp = 'Ŋ' THEN
    result:= result || '16';
  ELSIF tmp = 'ŋ' THEN
    result:= result || '16';
  ELSIF tmp = 'O' THEN
    result:= result || '17';
  ELSIF tmp = 'o' THEN
    result:= result || '17';
  ELSIF tmp = 'p' THEN
    result:= result || '18';
  ELSIF tmp = 'P' THEN
    result:= result || '18';
  ELSIF tmp = 'Q' THEN
    result:= result || '19';
  ELSIF tmp = 'q' THEN
    result:= result || '19';
  ELSIF tmp = 'R' THEN
    result:= result || '20';
  ELSIF tmp = 'r' THEN
    result:= result || '20';
  ELSIF tmp = 'S' THEN
    result:= result || '21';
  ELSIF tmp = 's' THEN
    result:= result || '21';
  ELSIF tmp = 'Š' THEN
    result:= result || '22';
  ELSIF tmp = 'š' THEN
    result:= result || '22';
  ELSIF tmp = 'T' THEN
    result:= result || '23';
  ELSIF tmp = 't' THEN
    result:= result || '23';
  ELSIF tmp = 'U' THEN
    result:= result || '24';
  ELSIF tmp = 'u' THEN
    result:= result || '24';
  ELSIF tmp = 'V' THEN
    result:= result || '25';
  ELSIF tmp = 'v' THEN
    result:= result || '25';
  ELSIF tmp = 'W' THEN
    result:= result || '26';
  ELSIF tmp = 'w' THEN
    result:= result || '26';
  ELSIF tmp = 'X' THEN
    result:= result || '27';
  ELSIF tmp = 'x' THEN
    result:= result || '27';
  ELSIF tmp = 'Y' THEN
    result:= result || '28';
  ELSIF tmp = 'y' THEN
    result:= result || '28';
  ELSIF tmp = 'Z' THEN
    result:= result || '29';
  ELSIF tmp = 'z' THEN
    result:= result || '29';
  ELSIF tmp = 'Ž' THEN
    result:= result || '30';
  ELSIF tmp = 'ž' THEN
    result:= result || '30';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- eng English
--
-- for the moment do nothing
CREATE OR REPLACE FUNCTION eng_sort( varchar ) 
 RETURNS varchar AS $PROC$

 DECLARE
  tmp		char;
  result	varchar := '''';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- est Estonian
-- a b c d e f g h i j k l m n o p q r s š z ž t u v w õ ä ö ü x y

CREATE OR REPLACE FUNCTION est_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	result := complete_numbers($1);
 	IF result = '' THEN
 	length = char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = '0'  THEN
    result:= result || '00';
  ELSIF tmp = '1'  THEN
    result:= result || '01';
  ELSIF tmp = '2'  THEN
    result:= result || '02';
  ELSIF tmp = '3'  THEN
    result:= result || '03';
  ELSIF tmp = '4'  THEN
    result:= result || '04';
  ELSIF tmp = '5'  THEN
    result:= result || '05';
  ELSIF tmp = '6'  THEN
    result:= result || '06';
  ELSIF tmp = '7'  THEN
    result:= result || '07';
  ELSIF tmp = '8'  THEN
    result:= result || '08';
  ELSIF tmp = '9'  THEN
    result:= result || '09';
  ELSIF tmp = 'A'  THEN
    result:= result || '21';
  ELSIF tmp = 'a'  THEN
    result:= result || '21';
  ELSIF tmp = 'À'  THEN
    result:= result || '21';
  ELSIF tmp = 'à'  THEN
    result:= result || '21';
  ELSIF tmp = 'B' THEN
    result:= result || '22';
  ELSIF tmp = 'b' THEN
    result:= result || '22';
  ELSIF tmp = 'C' THEN
    result:= result || '23';
  ELSIF tmp = 'c' THEN
    result:= result || '23';
  ELSIF tmp = 'D' THEN
    result:= result || '24';
  ELSIF tmp = 'd' THEN
    result:= result || '24';
  ELSIF tmp = 'e' THEN
    result:= result || '25';
  ELSIF tmp = 'E' THEN
    result:= result || '25';
  ELSIF tmp = 'F' THEN
    result:= result || '26';
  ELSIF tmp = 'f' THEN
    result:= result || '26';
  ELSIF tmp = 'G' THEN
    result:= result || '27';
  ELSIF tmp = 'g' THEN
    result:= result || '27';
  ELSIF tmp = 'H' THEN
    result:= result || '28';
  ELSIF tmp = 'h' THEN
    result:= result || '28';
  ELSIF tmp = 'I' THEN
    result:= result || '29';
  ELSIF tmp = 'i' THEN
    result:= result || '29';
  ELSIF tmp = 'J' THEN
    result:= result || '30';
  ELSIF tmp = 'j' THEN
    result:= result || '30';
  ELSIF tmp = 'K' THEN
    result:= result || '31';
  ELSIF tmp = 'k' THEN
    result:= result || '31';
  ELSIF tmp = 'L' THEN
    result:= result || '32';
  ELSIF tmp = 'l' THEN
    result:= result || '32';
  ELSIF tmp = 'M' THEN
    result:= result || '33';
  ELSIF tmp = 'm' THEN
    result:= result || '33';
  ELSIF tmp = 'N' THEN
    result:= result || '34';
  ELSIF tmp = 'n' THEN
    result:= result || '34';
  ELSIF tmp = 'O' THEN
    result:= result || '35';
  ELSIF tmp = 'o' THEN
    result:= result || '35';
  ELSIF tmp = 'P' THEN
    result:= result || '36';
  ELSIF tmp = 'p' THEN
    result:= result || '36';
  ELSIF tmp = 'Q' THEN
    result:= result || '37';
  ELSIF tmp = 'q' THEN
    result:= result || '37';
  ELSIF tmp = 'R' THEN
    result:= result || '38';
  ELSIF tmp = 'r' THEN
    result:= result || '38';
  ELSIF tmp = 'S' THEN
    result:= result || '39';
  ELSIF tmp = 's' THEN
    result:= result || '39';
  ELSIF tmp = 'Š' THEN
    result:= result || '40';
  ELSIF tmp = 'š' THEN
    result:= result || '40';
  ELSIF tmp = 'Z' THEN
    result:= result || '42';
  ELSIF tmp = 'z' THEN
    result:= result || '42';
  ELSIF tmp = 'Ž' THEN
    result:= result || '43';
  ELSIF tmp = 'ž' THEN
    result:= result || '43';
  ELSIF tmp = 'T' THEN
    result:= result || '44';
  ELSIF tmp = 't' THEN
    result:= result || '44';
  ELSIF tmp = 'U' THEN
    result:= result || '45';
  ELSIF tmp = 'u' THEN
    result:= result || '45';
  ELSIF tmp = 'V' THEN
    result:= result || '46';
  ELSIF tmp = 'v' THEN
    result:= result || '46';
  ELSIF tmp = 'W' THEN
    result:= result || '47';
  ELSIF tmp = 'w' THEN
    result:= result || '47';
  ELSIF tmp = 'Õ' THEN
    result:= result || '48';
  ELSIF tmp = 'õ' THEN
    result:= result || '48';
  ELSIF tmp = 'Ä' THEN
    result:= result || '49';
  ELSIF tmp = 'ä' THEN
    result:= result || '49';
  ELSIF tmp = 'Ö' THEN
    result:= result || '50';
  ELSIF tmp = 'ö' THEN
    result:= result || '50';
  ELSIF tmp = 'Ü' THEN
    result:= result || '51';
  ELSIF tmp = 'ü' THEN
    result:= result || '51';
  ELSIF tmp = 'X' THEN
    result:= result || '52';
  ELSIF tmp = 'x' THEN
    result:= result || '52';
  ELSIF tmp = 'Y' THEN
    result:= result || '53';
  ELSIF tmp = 'y' THEN
    result:= result || '53';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = $$'$$ THEN
    result:= result;
  ELSIF tmp = $$'$$ THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  END IF;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );



-- fra French
-- (a à ä â) b (c ç) d (e é è ë ê) f g h (i ï î) j k l m n (o ö ô) p q r s t (u ù ü û) v w x y z

CREATE OR REPLACE FUNCTION fra_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
 i		integer;
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'À' THEN
    result:= result || '01';
  ELSIF tmp = 'à' THEN
    result:= result || '01';
  ELSIF tmp = 'Â' THEN
    result:= result || '01';
  ELSIF tmp = 'â' THEN
    result:= result || '01';
  ELSIF tmp = 'Ä' THEN
    result:= result || '01';
  ELSIF tmp = 'ä' THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'C' THEN
    result:= result || '03';
  ELSIF tmp = 'c' THEN
    result:= result || '03';
  ELSIF tmp = 'Ç' THEN
    result:= result || '03';
  ELSIF tmp = 'ç' THEN
    result:= result || '03';
  ELSIF tmp = 'D' THEN
    result:= result || '04';
  ELSIF tmp = 'd' THEN
    result:= result || '04';
  ELSIF tmp = 'E' THEN
    result:= result || '05';
  ELSIF tmp = 'e' THEN
    result:= result || '05';
  ELSIF tmp = 'É' THEN
    result:= result || '05';
  ELSIF tmp = 'é' THEN
    result:= result || '05';
  ELSIF tmp = 'È' THEN
    result:= result || '05';
  ELSIF tmp = 'è' THEN
    result:= result || '05';
  ELSIF tmp = 'Ê' THEN
    result:= result || '05';
  ELSIF tmp = 'ê' THEN
    result:= result || '05';
  ELSIF tmp = 'Ë' THEN
    result:= result || '05';
  ELSIF tmp = 'ë' THEN
    result:= result || '05';
  ELSIF tmp = 'F' THEN
    result:= result || '06';
  ELSIF tmp = 'f' THEN
    result:= result || '06';
  ELSIF tmp = 'G' THEN
    result:= result || '07';
  ELSIF tmp = 'g' THEN
    result:= result || '07';
  ELSIF tmp = 'H' THEN
    result:= result || '08';
  ELSIF tmp = 'h' THEN
    result:= result || '08';
  ELSIF tmp = 'I' THEN
    result:= result || '09';
  ELSIF tmp = 'i' THEN
    result:= result || '09';
  ELSIF tmp = 'Î' THEN
    result:= result || '09';
  ELSIF tmp = 'î' THEN
    result:= result || '09';
  ELSIF tmp = 'Ï' THEN
    result:= result || '09';
  ELSIF tmp = 'ï' THEN
    result:= result || '09';
  ELSIF tmp = 'J' THEN
    result:= result || '10';
  ELSIF tmp = 'j' THEN
    result:= result || '10';
  ELSIF tmp = 'K' THEN
    result:= result || '11';
  ELSIF tmp = 'k' THEN
    result:= result || '11';
  ELSIF tmp = 'L' THEN
    result:= result || '12';
  ELSIF tmp = 'l' THEN
    result:= result || '12';
  ELSIF tmp = 'M' THEN
    result:= result || '13';
  ELSIF tmp = 'm' THEN
    result:= result || '13';
  ELSIF tmp = 'N' THEN
    result:= result || '14';
  ELSIF tmp = 'n' THEN
    result:= result || '14';
  ELSIF tmp = 'O' THEN
    result:= result || '15';
  ELSIF tmp = 'o' THEN
    result:= result || '15';
  ELSIF tmp = 'Ö' THEN
    result:= result || '15';
  ELSIF tmp = 'ö' THEN
    result:= result || '15';
  ELSIF tmp = 'ô' THEN
    result:= result || '15';
  ELSIF tmp = 'Ô' THEN
    result:= result || '15';
  ELSIF tmp = 'p' THEN
    result:= result || '16';
  ELSIF tmp = 'P' THEN
    result:= result || '16';
  ELSIF tmp = 'Q' THEN
    result:= result || '17';
  ELSIF tmp = 'q' THEN
    result:= result || '17';
  ELSIF tmp = 'R' THEN
    result:= result || '18';
  ELSIF tmp = 'r' THEN
    result:= result || '18';
  ELSIF tmp = 'S' THEN
    result:= result || '19';
  ELSIF tmp = 's' THEN
    result:= result || '19';
  ELSIF tmp = 'T' THEN
    result:= result || '20';
  ELSIF tmp = 't' THEN
    result:= result || '20';
  ELSIF tmp = 'U' THEN
    result:= result || '22';
  ELSIF tmp = 'u' THEN
    result:= result || '22';
  ELSIF tmp = 'Ù' THEN
    result:= result || '22';
  ELSIF tmp = 'ù' THEN
    result:= result || '22';
  ELSIF tmp = 'Û' THEN
    result:= result || '22';
  ELSIF tmp = 'û' THEN
    result:= result || '22';
  ELSIF tmp = 'Ü' THEN
    result:= result || '22';
  ELSIF tmp = 'ü' THEN
    result:= result || '22';
  ELSIF tmp = 'V' THEN
    result:= result || '23';
  ELSIF tmp = 'v' THEN
    result:= result || '23';
  ELSIF tmp = 'W' THEN
    result:= result || '24';
  ELSIF tmp = 'w' THEN
    result:= result || '24';
  ELSIF tmp = 'X' THEN
    result:= result || '25';
  ELSIF tmp = 'x' THEN
    result:= result || '25';
  ELSIF tmp = 'Y' THEN
    result:= result || '26';
  ELSIF tmp = 'y' THEN
    result:= result || '26';
  ELSIF tmp = 'Z' THEN
    result:= result || '27';
  ELSIF tmp = 'z' THEN
    result:= result || '27';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );



-- ful Fulfulde
-- a b mb ɓ c d nd ɗ e f  g  ng h  i  j  ǌ  k  l  m  n  ny ŋ  o  p  r  s  t  u  w  y  ƴ
-- 1 2 3  4 5 6 7  8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

CREATE OR REPLACE FUNCTION ful_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
  i		integer := 1;
  tmp		char;
  tmp2		char = SUBSTR( $1, 1, 1 );
  tmp3		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	WHILE i <= length LOOP
		tmp := tmp2;
		tmp2 := SUBSTR( $1, i+1, 1 );
		tmp3 := SUBSTR( $1, i+2, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'Ɓ' THEN
    result:= result || '04';
  ELSIF tmp = 'ɓ' THEN
    result:= result || '04';
  ELSIF tmp = 'C' THEN
    result:= result || '05';
  ELSIF tmp = 'c' THEN
    result:= result || '05';
  ELSIF tmp = 'D' THEN
    result:= result || '06';
  ELSIF tmp = 'd' THEN
    result:= result || '06';	
  ELSIF tmp = 'Ɗ' THEN
    result:= result || '08';
  ELSIF tmp = 'ɗ' THEN
    result:= result || '08';
  ELSIF tmp = 'E' THEN
    result:= result || '09';
  ELSIF tmp = 'e' THEN
    result:= result || '09';
  ELSIF tmp = 'F' THEN
	result:= result || '10';
  ELSIF tmp = 'f' THEN
	result:= result || '10';
  ELSIF tmp = 'G' THEN
	result:= result || '11';
  ELSIF tmp = 'g' THEN
	result:= result || '11';
  ELSIF tmp = 'H' THEN
    result:= result || '13';
  ELSIF tmp = 'h' THEN
    result:= result || '13';
  ELSIF tmp = 'I' THEN
    result:= result || '14';
  ELSIF tmp = 'i' THEN
    result:= result || '14';
  ELSIF tmp = 'J' THEN
    result:= result || '15';
  ELSIF tmp = 'j' THEN
    result:= result || '15';
  ELSIF tmp = 'ǋ' THEN
    result:= result || '16';
  ELSIF tmp = 'ǌ' THEN
    result:= result || '16';
  ELSIF tmp = 'K' THEN
	result:= result || '17';
  ELSIF tmp = 'k' THEN
	result:= result || '17';
  ELSIF tmp = 'L' THEN
    result:= result || '18';
  ELSIF tmp = 'l' THEN
    result:= result || '18';
  ELSIF tmp = 'M' THEN
	IF tmp2 = 'B' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '03';
	ELSIF tmp2 = 'b' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '03';
	ELSE
		result:= result || '19';
	END IF;
  ELSIF tmp = 'm' THEN
	IF tmp2 = 'b' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '03';
	ELSE
		result:= result || '19';
	END IF;
  ELSIF tmp = 'N' THEN
	IF tmp2 = 'D' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '07';
	ELSIF tmp2 = 'd' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '07';
	ELSIF tmp2 = 'G' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '12';
	ELSIF tmp2 = 'g' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '12';
	ELSIF tmp2 = 'J' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '16';
	ELSIF tmp2 = 'j' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '16';
	ELSEIF tmp2 = 'Y' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '21';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '21';
	ELSE
		result:= result || '20';
	END IF;
  ELSIF tmp = 'n' THEN
	IF tmp2 = 'd' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '07';
	ELSIF tmp2 = 'g' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '12';
	ELSIF tmp2 = 'j' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '16';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '21';
	ELSE
		result:= result || '20';
	END IF;
 ELSIF tmp = 'Ŋ' THEN
    result:= result || '22';
  ELSIF tmp = 'ŋ' THEN
    result:= result || '22';
 ELSIF tmp = 'O' THEN
    result:= result || '23';
  ELSIF tmp = 'o' THEN
    result:= result || '23';
  ELSIF tmp = 'p' THEN
    result:= result || '24';
  ELSIF tmp = 'P' THEN
    result:= result || '24';
  ELSIF tmp = 'R' THEN
    result:= result || '25';
  ELSIF tmp = 'r' THEN
    result:= result || '25';
 ELSIF tmp = 'S' THEN
	result:= result || '26';
  ELSIF tmp = 's' THEN
	result:= result || '26';
 ELSIF tmp = 'T' THEN
	result:= result || '27';
  ELSIF tmp = 't' THEN
	result:= result || '27';
  ELSIF tmp = 'U' THEN
    result:= result || '28';
  ELSIF tmp = 'u' THEN
    result:= result || '28';
  ELSIF tmp = 'W' THEN
    result:= result || '29';
  ELSIF tmp = 'w' THEN
    result:= result || '29';
  ELSIF tmp = 'Y' THEN
    result:= result || '30';
  ELSIF tmp = 'y' THEN
    result:= result || '30';
  ELSIF tmp = 'Ƴ' THEN
    result:= result || '31';
  ELSIF tmp = 'ƴ' THEN
    result:= result || '31';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
  i:=i+1;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- hau Hausa
-- a b ɓ c d ɗ e f fy g gw gy h i j k kw ky ƙ ƙw ƙy l m n o p r s sh t ts u w y ƴ z

CREATE OR REPLACE FUNCTION hau_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
  i		integer := 1;
  tmp		char;
  tmp2		char = SUBSTR( $1, 1, 1 );
  tmp3		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	WHILE i <= length LOOP
		tmp := tmp2;
		tmp2 := SUBSTR( $1, i+1, 1 );
		tmp3 := SUBSTR( $1, i+2, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'Ɓ' THEN
    result:= result || '03';
  ELSIF tmp = 'ɓ' THEN
    result:= result || '03';
  ELSIF tmp = 'C' THEN
    result:= result || '04';
  ELSIF tmp = 'c' THEN
    result:= result || '04';
  ELSIF tmp = 'D' THEN
    result:= result || '05';
  ELSIF tmp = 'd' THEN
    result:= result || '05';
  ELSIF tmp = 'Ɗ' THEN
    result:= result || '06';
  ELSIF tmp = 'ɗ' THEN
    result:= result || '06';
  ELSIF tmp = 'E' THEN
    result:= result || '07';
  ELSIF tmp = 'e' THEN
    result:= result || '07';
  ELSIF tmp = 'F' THEN
	IF tmp2 = 'Y' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '09';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '09';
	ELSE
		result:= result || '08';
	END IF;
  ELSIF tmp = 'f' THEN
	IF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '09';
	ELSE
		result:= result || '08';
	END IF;
  ELSIF tmp = 'G' THEN
	IF tmp2 = 'W' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '11';
	ELSIF tmp2 = 'w' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '11';
	ELSIF tmp2 = 'Y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '12';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '12';
	ELSE
		result:= result || '10';
	END IF;
  ELSIF tmp = 'g' THEN
	IF tmp2 = 'w' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '11';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '12';
	ELSE
		result:= result || '10';
	END IF;
  ELSIF tmp = 'H' THEN
    result:= result || '13';
  ELSIF tmp = 'h' THEN
    result:= result || '13';
  ELSIF tmp = 'I' THEN
    result:= result || '14';
  ELSIF tmp = 'i' THEN
    result:= result || '14';
  ELSIF tmp = 'J' THEN
    result:= result || '15';
  ELSIF tmp = 'j' THEN
    result:= result || '15';
  ELSIF tmp = 'K' THEN
	IF tmp2 = 'W' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '17';
	ELSIF tmp2 = 'w' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '17';
	ELSIF tmp2 = 'Y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '18';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '18';
	ELSE
		result:= result || '16';
	END IF;
  ELSIF tmp = 'k' THEN
	IF tmp2 = 'w' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '17';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '18';
	ELSE
		result:= result || '16';
	END IF;
  ELSIF tmp = 'Ƙ' THEN
	IF tmp2 = 'W' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '20';
	ELSIF tmp2 = 'w' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '20';
	ELSIF tmp2 = 'Y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '21';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '21';
	ELSE
		result:= result || '19';
	END IF;
  ELSIF tmp = 'ƙ' THEN
	IF tmp2 = 'w' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '20';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '21';
	ELSE
		result:= result || '19';
	END IF;
  ELSIF tmp = 'L' THEN
    result:= result || '22';
  ELSIF tmp = 'l' THEN
    result:= result || '22';
  ELSIF tmp = 'M' THEN
    result:= result || '23';
  ELSIF tmp = 'm' THEN
    result:= result || '23';
  ELSIF tmp = 'N' THEN
    result:= result || '24';
  ELSIF tmp = 'n' THEN
    result:= result || '24';
  ELSIF tmp = 'O' THEN
    result:= result || '25';
  ELSIF tmp = 'o' THEN
    result:= result || '25';
  ELSIF tmp = 'p' THEN
    result:= result || '26';
  ELSIF tmp = 'P' THEN
    result:= result || '26';
  ELSIF tmp = 'R' THEN
    result:= result || '27';
  ELSIF tmp = 'r' THEN
    result:= result || '27';
  ELSIF tmp = 'S' THEN
	IF tmp2 = 'H' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '29';
	ELSIF tmp2 = 'h' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '29';
	ELSE
		result:= result || '28';
	END IF;
  ELSIF tmp = 's' THEN
	IF tmp2 = 'h' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '29';
	ELSE
		result:= result || '28';
	END IF;
  ELSIF tmp = 'T' THEN
	IF tmp2 = 'S' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '31';
	ELSIF tmp2 = 's' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '31';
	ELSE
		result:= result || '30';
	END IF;
  ELSIF tmp = 't' THEN
	IF tmp2 = 's' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '31';
	ELSE
		result:= result || '30';
	END IF;
  ELSIF tmp = 'ʦ' THEN
    result:= result || '31';
  ELSIF tmp = 'U' THEN
    result:= result || '32';
  ELSIF tmp = 'u' THEN
    result:= result || '32';
  ELSIF tmp = 'W' THEN
    result:= result || '33';
  ELSIF tmp = 'w' THEN
    result:= result || '33';
  ELSIF tmp = 'Y' THEN
    result:= result || '34';
  ELSIF tmp = 'y' THEN
    result:= result || '34';
  ELSIF tmp = 'Ƴ' THEN
    result:= result || '35';
  ELSIF tmp = 'Ƴ' THEN
    result:= result || '35';
  ELSIF tmp = 'Z' THEN
    result:= result || '36';
  ELSIF tmp = 'z' THEN
    result:= result || '36';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
  i:=i+1;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- ita Italian
-- (a à) b c d (e é è) f g h (i ï ì í) j k l m n (o ò ó Ó Ò) p q r s t (u Ù ù Ú ú) v w x y z

CREATE OR REPLACE FUNCTION ita_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
 i		integer;
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'À' THEN
    result:= result || '01';
  ELSIF tmp = 'à' THEN
    result:= result || '01';
  ELSIF tmp = 'Â' THEN
    result:= result || '01';
  ELSIF tmp = 'â' THEN
    result:= result || '01';
  ELSIF tmp = 'Ä' THEN
    result:= result || '01';
  ELSIF tmp = 'ä' THEN
    result:= result || '01';
  ELSIF tmp = 'Á' THEN
    result:= result || '01';
  ELSIF tmp = 'á' THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'C' THEN
    result:= result || '03';
  ELSIF tmp = 'c' THEN
    result:= result || '03';
  ELSIF tmp = 'Ç' THEN
    result:= result || '03';
  ELSIF tmp = 'ç' THEN
    result:= result || '03';
  ELSIF tmp = 'D' THEN
    result:= result || '04';
  ELSIF tmp = 'd' THEN
    result:= result || '04';
  ELSIF tmp = 'E' THEN
    result:= result || '05';
  ELSIF tmp = 'e' THEN
    result:= result || '05';
  ELSIF tmp = 'É' THEN
    result:= result || '05';
  ELSIF tmp = 'é' THEN
    result:= result || '05';
  ELSIF tmp = 'È' THEN
    result:= result || '05';
  ELSIF tmp = 'è' THEN
    result:= result || '05';
  ELSIF tmp = 'Ê' THEN
    result:= result || '05';
  ELSIF tmp = 'ê' THEN
    result:= result || '05';
  ELSIF tmp = 'Ë' THEN
    result:= result || '05';
  ELSIF tmp = 'ë' THEN
    result:= result || '05';
  ELSIF tmp = 'F' THEN
    result:= result || '06';
  ELSIF tmp = 'f' THEN
    result:= result || '06';
  ELSIF tmp = 'G' THEN
    result:= result || '07';
  ELSIF tmp = 'g' THEN
    result:= result || '07';
  ELSIF tmp = 'H' THEN
    result:= result || '08';
  ELSIF tmp = 'h' THEN
    result:= result || '08';
  ELSIF tmp = 'I' THEN
    result:= result || '09';
  ELSIF tmp = 'i' THEN
    result:= result || '09';
  ELSIF tmp = 'Î' THEN
    result:= result || '09';
  ELSIF tmp = 'î' THEN
    result:= result || '09';
  ELSIF tmp = 'Ï' THEN
    result:= result || '09';
  ELSIF tmp = 'ï' THEN
    result:= result || '09';
  ELSIF tmp = 'Ì' THEN
    result:= result || '09';
  ELSIF tmp = 'ì' THEN
    result:= result || '09';
  ELSIF tmp = 'Í' THEN
    result:= result || '09';
  ELSIF tmp = 'í' THEN
    result:= result || '09';
  ELSIF tmp = 'J' THEN
    result:= result || '10';
  ELSIF tmp = 'j' THEN
    result:= result || '10';
  ELSIF tmp = 'K' THEN
    result:= result || '11';
  ELSIF tmp = 'k' THEN
    result:= result || '11';
  ELSIF tmp = 'L' THEN
    result:= result || '12';
  ELSIF tmp = 'l' THEN
    result:= result || '12';
  ELSIF tmp = 'M' THEN
    result:= result || '13';
  ELSIF tmp = 'm' THEN
    result:= result || '13';
  ELSIF tmp = 'N' THEN
    result:= result || '14';
  ELSIF tmp = 'n' THEN
    result:= result || '14';
  ELSIF tmp = 'O' THEN
    result:= result || '15';
  ELSIF tmp = 'o' THEN
    result:= result || '15';
  ELSIF tmp = 'Ö' THEN
    result:= result || '15';
  ELSIF tmp = 'ö' THEN
    result:= result || '15';
  ELSIF tmp = 'ô' THEN
    result:= result || '15';
  ELSIF tmp = 'Ô' THEN
    result:= result || '15';
  ELSIF tmp = 'Ó' THEN
    result:= result || '15';
  ELSIF tmp = 'ó' THEN
    result:= result || '15';
  ELSIF tmp = 'Ò' THEN
    result:= result || '15';
  ELSIF tmp = 'ò' THEN
    result:= result || '15';
  ELSIF tmp = 'p' THEN
    result:= result || '16';
  ELSIF tmp = 'P' THEN
    result:= result || '16';
  ELSIF tmp = 'Q' THEN
    result:= result || '17';
  ELSIF tmp = 'q' THEN
    result:= result || '17';
  ELSIF tmp = 'R' THEN
    result:= result || '18';
  ELSIF tmp = 'r' THEN
    result:= result || '18';
  ELSIF tmp = 'S' THEN
    result:= result || '19';
  ELSIF tmp = 's' THEN
    result:= result || '19';
  ELSIF tmp = 'T' THEN
    result:= result || '20';
  ELSIF tmp = 't' THEN
    result:= result || '20';
  ELSIF tmp = 'U' THEN
    result:= result || '22';
  ELSIF tmp = 'u' THEN
    result:= result || '22';
  ELSIF tmp = 'Ù' THEN
    result:= result || '22';
  ELSIF tmp = 'ù' THEN
    result:= result || '22';
  ELSIF tmp = 'Û' THEN
    result:= result || '22';
  ELSIF tmp = 'û' THEN
    result:= result || '22';
  ELSIF tmp = 'Ü' THEN
    result:= result || '22';
  ELSIF tmp = 'ü' THEN
    result:= result || '22';
  ELSIF tmp = 'Ú' THEN
    result:= result || '22';
  ELSIF tmp = 'ú' THEN
    result:= result || '22';
  ELSIF tmp = 'V' THEN
    result:= result || '23';
  ELSIF tmp = 'v' THEN
    result:= result || '23';
  ELSIF tmp = 'W' THEN
    result:= result || '24';
  ELSIF tmp = 'w' THEN
    result:= result || '24';
  ELSIF tmp = 'X' THEN
    result:= result || '25';
  ELSIF tmp = 'x' THEN
    result:= result || '25';
  ELSIF tmp = 'Y' THEN
    result:= result || '26';
  ELSIF tmp = 'y' THEN
    result:= result || '26';
  ELSIF tmp = 'Z' THEN
    result:= result || '27';
  ELSIF tmp = 'z' THEN
    result:= result || '27';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- jpn Japanese
--
-- romaji katakana hiragana kanji
-- hiragana and katakana are sorted with the gojûon order. 
-- あいうえお かきくけこ さしすせそ たちつてと なにぬねの はひふへほ まみむめも らりるれろ やゆよ わをん
-- This sort gives the correct sort:
-- すす,　すず,　すすき,　すすぎ,　すずき,　すすむ,　すずむ　(susu, suzu, susuki, susugi, suzuki, susumu, suzumu)
-- what to do with the ー　?
-- select word collate "ja_JP" from test order by word asc gives a wrong sort:
--  すす, すすき, すすぎ, すすむ, すず, すずき, すずむ

CREATE OR REPLACE FUNCTION jpn_sort( varchar ) 
 RETURNS varchar AS $PROC$

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
  BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  	  tmp := SUBSTR( $1, i, 1 );
	  IF tmp = 'あ'  THEN
		result:= result || '11';
	  ELSIF tmp = 'ア'  THEN
		result:= result || '11';
	  ELSIF tmp = 'ぁ'  THEN
		result:= result || '11';
	  ELSIF tmp = 'ァ'  THEN
		result:= result || '11';
	  ELSIF tmp = 'い' THEN
		result:= result || '12';
	  ELSIF tmp = 'イ' THEN
		result:= result || '12';
	  ELSIF tmp = 'ぃ' THEN
		result:= result || '12';
	  ELSIF tmp = 'ィ' THEN
		result:= result || '12';
	  ELSIF tmp = 'う' THEN
		result:= result || '13';
	  ELSIF tmp = 'ウ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ぅ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ゥ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ゔ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ヴ' THEN
		result:= result || '13';
	  ELSIF tmp = 'え' THEN
		result:= result || '14';
	  ELSIF tmp = 'エ' THEN
		result:= result || '14';
	  ELSIF tmp = 'ぇ' THEN
		result:= result || '14';
	  ELSIF tmp = 'ェ' THEN
		result:= result || '14';
	  ELSIF tmp = 'お' THEN
		result:= result || '15';
	  ELSIF tmp = 'オ' THEN
		result:= result || '15';
	  ELSIF tmp = 'ぉ' THEN
		result:= result || '15';
	  ELSIF tmp = 'ォ' THEN
		result:= result || '15';
	  ELSIF tmp = 'か' THEN
		result:= result || '16';
	  ELSIF tmp = 'カ' THEN
		result:= result || '16';
	  ELSIF tmp = 'が' THEN
		result:= result || '16';
	  ELSIF tmp = 'ガ' THEN
		result:= result || '16';
	  ELSIF tmp = 'き' THEN
		result:= result || '18';
	  ELSIF tmp = 'キ' THEN
		result:= result || '18';
	  ELSIF tmp = 'ぎ' THEN
		result:= result || '18';
	  ELSIF tmp = 'ギ' THEN
		result:= result || '18';
	  ELSIF tmp = 'く' THEN
		result:= result || '20';
	  ELSIF tmp = 'ク' THEN
		result:= result || '20';
	  ELSIF tmp = 'ぐ' THEN
		result:= result || '20';
	  ELSIF tmp = 'グ' THEN
		result:= result || '20';
	  ELSIF tmp = 'け' THEN
		result:= result || '22';
	  ELSIF tmp = 'ケ' THEN
		result:= result || '22';
	  ELSIF tmp = 'げ' THEN
		result:= result || '22';
	  ELSIF tmp = 'ゲ' THEN
		result:= result || '22';
	  ELSIF tmp = 'こ' THEN
		result:= result || '24';
	  ELSIF tmp = 'コ' THEN
		result:= result || '24';
	  ELSIF tmp = 'ご' THEN
		result:= result || '24';
	  ELSIF tmp = 'ゴ' THEN
		result:= result || '24';
	  ELSIF tmp = 'さ' THEN
		result:= result || '26';
	  ELSIF tmp = 'サ' THEN
		result:= result || '26';
	  ELSIF tmp = 'ざ' THEN
		result:= result || '26';
	  ELSIF tmp = 'ザ' THEN
		result:= result || '26';
	  ELSIF tmp = 'し' THEN
		result:= result || '28';
	  ELSIF tmp = 'シ' THEN
		result:= result || '28';
	  ELSIF tmp = 'じ' THEN
		result:= result || '28';
	  ELSIF tmp = 'ジ' THEN
		result:= result || '28';
	  ELSIF tmp = 'す' THEN
		result:= result || '30';
	  ELSIF tmp = 'ス' THEN
		result:= result || '30';
	  ELSIF tmp = 'ず' THEN
		result:= result || '30';
	  ELSIF tmp = 'ズ' THEN
		result:= result || '30';
	  ELSIF tmp = 'せ' THEN
		result:= result || '32';
	  ELSIF tmp = 'セ' THEN
		result:= result || '32';
	  ELSIF tmp = 'ぜ' THEN
		result:= result || '32';
	  ELSIF tmp = 'ゼ' THEN
		result:= result || '32';
	  ELSIF tmp = 'そ' THEN
		result:= result || '34';
	  ELSIF tmp = 'ソ' THEN
		result:= result || '34';
	  ELSIF tmp = 'ぞ' THEN
		result:= result || '34';
	  ELSIF tmp = 'ゾ' THEN
		result:= result || '34';
	  ELSIF tmp = 'た' THEN
		result:= result || '36';
	  ELSIF tmp = 'タ' THEN
		result:= result || '36';
	  ELSIF tmp = 'だ' THEN
		result:= result || '36';
	  ELSIF tmp = 'ダ' THEN
		result:= result || '36';
	  ELSIF tmp = 'ち' THEN
		result:= result || '38';
	  ELSIF tmp = 'チ' THEN
		result:= result || '38';
	  ELSIF tmp = 'ぢ' THEN
		result:= result || '38';
	  ELSIF tmp = 'ヂ' THEN
		result:= result || '38';
	  ELSIF tmp = 'つ' THEN
		result:= result || '40';
	  ELSIF tmp = 'ツ' THEN
		result:= result || '40';
	  ELSIF tmp = 'づ' THEN
		result:= result || '40';
	  ELSIF tmp = 'ヅ' THEN
		result:= result || '40';
	  ELSIF tmp = 'っ' THEN
		result:= result || '40';
	  ELSIF tmp = 'ッ' THEN
		result:= result || '40';
	  ELSIF tmp = 'て' THEN
		result:= result || '42';
	  ELSIF tmp = 'テ' THEN
		result:= result || '42';
	  ELSIF tmp = 'で' THEN
		result:= result || '42';
	  ELSIF tmp = 'デ' THEN
		result:= result || '42';
	  ELSIF tmp = 'と' THEN
		result:= result || '44';
	  ELSIF tmp = 'ト' THEN
		result:= result || '44';
	  ELSIF tmp = 'ど' THEN
		result:= result || '44';
	  ELSIF tmp = 'ド' THEN
		result:= result || '44';
	  ELSIF tmp = 'な' THEN
		result:= result || '46';
	  ELSIF tmp = 'ナ' THEN
		result:= result || '46';
	  ELSIF tmp = 'に' THEN
		result:= result || '47';
	  ELSIF tmp = 'ニ' THEN
		result:= result || '47';
	  ELSIF tmp = 'ぬ' THEN
		result:= result || '48';
	  ELSIF tmp = 'ヌ' THEN
		result:= result || '48';
	  ELSIF tmp = 'ね' THEN
		result:= result || '49';
	  ELSIF tmp = 'ネ' THEN
		result:= result || '49';
	  ELSIF tmp = 'の' THEN
		result:= result || '50';
	  ELSIF tmp = 'ノ' THEN
		result:= result || '50';
	  ELSIF tmp = 'は' THEN
		result:= result || '51';
	  ELSIF tmp = 'ハ' THEN
		result:= result || '51';
	  ELSIF tmp = 'ば' THEN
		result:= result || '51';
	  ELSIF tmp = 'バ' THEN
		result:= result || '51';
	  ELSIF tmp = 'ぱ' THEN
		result:= result || '51';
	  ELSIF tmp = 'パ' THEN
		result:= result || '51';
	  ELSIF tmp = 'ひ' THEN
		result:= result || '54';
	  ELSIF tmp = 'ヒ' THEN
		result:= result || '54';
	  ELSIF tmp = 'び' THEN
		result:= result || '54';
	  ELSIF tmp = 'ビ' THEN
		result:= result || '54';
	  ELSIF tmp = 'ぴ' THEN
		result:= result || '54';
	  ELSIF tmp = 'ピ' THEN
		result:= result || '54';
	  ELSIF tmp = 'ふ' THEN
		result:= result || '57';
	  ELSIF tmp = 'フ' THEN
		result:= result || '57';
	  ELSIF tmp = 'ぶ' THEN
		result:= result || '57';
	  ELSIF tmp = 'ブ' THEN
		result:= result || '57';
	  ELSIF tmp = 'ぷ' THEN
		result:= result || '57';
	  ELSIF tmp = 'プ' THEN
		result:= result || '57';
	  ELSIF tmp = 'へ' THEN
		result:= result || '60';
	  ELSIF tmp = 'ヘ' THEN
		result:= result || '60';
	  ELSIF tmp = 'べ' THEN
		result:= result || '60';
	  ELSIF tmp = 'ベ' THEN
		result:= result || '60';
	  ELSIF tmp = 'ぺ' THEN
		result:= result || '60';
	  ELSIF tmp = 'ペ' THEN
		result:= result || '60';
	  ELSIF tmp = 'ほ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ホ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ぼ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ボ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ぽ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ポ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ま' THEN
		result:= result || '66';
	  ELSIF tmp = 'マ' THEN
		result:= result || '66';
	  ELSIF tmp = 'み' THEN
		result:= result || '67';
	  ELSIF tmp = 'ミ' THEN
		result:= result || '67';
	  ELSIF tmp = 'む' THEN
		result:= result || '68';
	  ELSIF tmp = 'ム' THEN
		result:= result || '68';
	  ELSIF tmp = 'め' THEN
		result:= result || '69';
	  ELSIF tmp = 'メ' THEN
		result:= result || '69';
	  ELSIF tmp = 'も' THEN
		result:= result || '70';
	  ELSIF tmp = 'モ' THEN
		result:= result || '70';
	  ELSIF tmp = 'ら' THEN
		result:= result || '71';
	  ELSIF tmp = 'ラ' THEN
		result:= result || '71';
	  ELSIF tmp = 'り' THEN
		result:= result || '72';
	  ELSIF tmp = 'リ' THEN
		result:= result || '72';
	  ELSIF tmp = 'る' THEN
		result:= result || '73';
	  ELSIF tmp = 'ル' THEN
		result:= result || '73';
	  ELSIF tmp = 'れ' THEN
		result:= result || '74';
	  ELSIF tmp = 'レ' THEN
		result:= result || '74';
	  ELSIF tmp = 'ろ' THEN
		result:= result || '75';
	  ELSIF tmp = 'ロ' THEN
		result:= result || '75';
	  ELSIF tmp = 'や' THEN
		result:= result || '76';
	  ELSIF tmp = 'ヤ' THEN
		result:= result || '76';
	  ELSIF tmp = 'ゃ' THEN
		result:= result || '76';
	  ELSIF tmp = 'ャ' THEN
		result:= result || '76';
	  ELSIF tmp = 'ゆ' THEN
		result:= result || '77';
	  ELSIF tmp = 'ユ' THEN
		result:= result || '77';
	  ELSIF tmp = 'ゅ' THEN
		result:= result || '77';
	  ELSIF tmp = 'ュ' THEN
		result:= result || '77';
	  ELSIF tmp = 'よ' THEN
		result:= result || '78';
	  ELSIF tmp = 'ヨ' THEN
		result:= result || '78';
	  ELSIF tmp = 'ょ' THEN
		result:= result || '78';
	  ELSIF tmp = 'ョ' THEN
		result:= result || '78';
	  ELSIF tmp = 'わ' THEN
		result:= result || '79';
	  ELSIF tmp = 'ワ' THEN
		result:= result || '79';
	  ELSIF tmp = 'ゐ' THEN
		result:= result || '80';
	  ELSIF tmp = 'ヰ' THEN
		result:= result || '80';
	  ELSIF tmp = 'ゑ' THEN
		result:= result || '81';
	  ELSIF tmp = 'ヱ' THEN
		result:= result || '81';
	  ELSIF tmp = 'を' THEN
		result:= result || '82';
	  ELSIF tmp = 'ヲ' THEN
		result:= result || '82';
	  ELSIF tmp = 'ん' THEN
		result:= result || '83';
	  ELSIF tmp = 'ン' THEN
		result:= result || '83';
	  ELSIF tmp = 'ー' THEN
		result:= result;
	  ELSE
		result:= result || tmp;
	  END IF;
  	END LOOP;
  	result:= result || '00';
  	FOR i IN 1.. length LOOP
	  tmp := SUBSTR( $1, i, 1 );
	  IF tmp = 'あ'  THEN
		result:= result || '11';
	  ELSIF tmp = 'ア'  THEN
		result:= result || '11';
	  ELSIF tmp = 'ぁ'  THEN
		result:= result || '11';
	  ELSIF tmp = 'ァ'  THEN
		result:= result || '11';
	  ELSIF tmp = 'い' THEN
		result:= result || '12';
	  ELSIF tmp = 'イ' THEN
		result:= result || '12';
	  ELSIF tmp = 'ぃ' THEN
		result:= result || '12';
	  ELSIF tmp = 'ィ' THEN
		result:= result || '12';
	  ELSIF tmp = 'う' THEN
		result:= result || '13';
	  ELSIF tmp = 'ウ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ぅ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ゥ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ゔ' THEN
		result:= result || '13';
	  ELSIF tmp = 'ヴ' THEN
		result:= result || '13';
	  ELSIF tmp = 'え' THEN
		result:= result || '14';
	  ELSIF tmp = 'エ' THEN
		result:= result || '14';
	  ELSIF tmp = 'ぇ' THEN
		result:= result || '14';
	  ELSIF tmp = 'ェ' THEN
		result:= result || '14';
	  ELSIF tmp = 'お' THEN
		result:= result || '15';
	  ELSIF tmp = 'オ' THEN
		result:= result || '15';
	  ELSIF tmp = 'ぉ' THEN
		result:= result || '15';
	  ELSIF tmp = 'ォ' THEN
		result:= result || '15';
	  ELSIF tmp = 'か' THEN
		result:= result || '16';
	  ELSIF tmp = 'カ' THEN
		result:= result || '16';
	  ELSIF tmp = 'が' THEN
		result:= result || '17';
	  ELSIF tmp = 'ガ' THEN
		result:= result || '17';
	  ELSIF tmp = 'き' THEN
		result:= result || '18';
	  ELSIF tmp = 'キ' THEN
		result:= result || '18';
	  ELSIF tmp = 'ぎ' THEN
		result:= result || '19';
	  ELSIF tmp = 'ギ' THEN
		result:= result || '19';
	  ELSIF tmp = 'く' THEN
		result:= result || '20';
	  ELSIF tmp = 'ク' THEN
		result:= result || '20';
	  ELSIF tmp = 'ぐ' THEN
		result:= result || '21';
	  ELSIF tmp = 'グ' THEN
		result:= result || '21';
	  ELSIF tmp = 'け' THEN
		result:= result || '22';
	  ELSIF tmp = 'ケ' THEN
		result:= result || '22';
	  ELSIF tmp = 'げ' THEN
		result:= result || '23';
	  ELSIF tmp = 'ゲ' THEN
		result:= result || '23';
	  ELSIF tmp = 'こ' THEN
		result:= result || '24';
	  ELSIF tmp = 'コ' THEN
		result:= result || '24';
	  ELSIF tmp = 'ご' THEN
		result:= result || '25';
	  ELSIF tmp = 'ゴ' THEN
		result:= result || '25';
	  ELSIF tmp = 'さ' THEN
		result:= result || '26';
	  ELSIF tmp = 'サ' THEN
		result:= result || '26';
	  ELSIF tmp = 'ざ' THEN
		result:= result || '27';
	  ELSIF tmp = 'ザ' THEN
		result:= result || '27';
	  ELSIF tmp = 'し' THEN
		result:= result || '28';
	  ELSIF tmp = 'シ' THEN
		result:= result || '28';
	  ELSIF tmp = 'じ' THEN
		result:= result || '29';
	  ELSIF tmp = 'ジ' THEN
		result:= result || '29';
	  ELSIF tmp = 'す' THEN
		result:= result || '30';
	  ELSIF tmp = 'ス' THEN
		result:= result || '30';
	  ELSIF tmp = 'ず' THEN
		result:= result || '31';
	  ELSIF tmp = 'ズ' THEN
		result:= result || '31';
	  ELSIF tmp = 'せ' THEN
		result:= result || '32';
	  ELSIF tmp = 'セ' THEN
		result:= result || '32';
	  ELSIF tmp = 'ぜ' THEN
		result:= result || '33';
	  ELSIF tmp = 'ゼ' THEN
		result:= result || '33';
	  ELSIF tmp = 'そ' THEN
		result:= result || '34';
	  ELSIF tmp = 'ソ' THEN
		result:= result || '34';
	  ELSIF tmp = 'ぞ' THEN
		result:= result || '35';
	  ELSIF tmp = 'ゾ' THEN
		result:= result || '35';
	  ELSIF tmp = 'た' THEN
		result:= result || '36';
	  ELSIF tmp = 'タ' THEN
		result:= result || '36';
	  ELSIF tmp = 'だ' THEN
		result:= result || '37';
	  ELSIF tmp = 'ダ' THEN
		result:= result || '37';
	  ELSIF tmp = 'ち' THEN
		result:= result || '38';
	  ELSIF tmp = 'チ' THEN
		result:= result || '38';
	  ELSIF tmp = 'ぢ' THEN
		result:= result || '39';
	  ELSIF tmp = 'ヂ' THEN
		result:= result || '39';
	  ELSIF tmp = 'つ' THEN
		result:= result || '40';
	  ELSIF tmp = 'ツ' THEN
		result:= result || '40';
	  ELSIF tmp = 'づ' THEN
		result:= result || '41';
	  ELSIF tmp = 'ヅ' THEN
		result:= result || '41';
	  ELSIF tmp = 'っ' THEN
		result:= result || '40';
	  ELSIF tmp = 'ッ' THEN
		result:= result || '40';
	  ELSIF tmp = 'て' THEN
		result:= result || '42';
	  ELSIF tmp = 'テ' THEN
		result:= result || '42';
	  ELSIF tmp = 'で' THEN
		result:= result || '43';
	  ELSIF tmp = 'デ' THEN
		result:= result || '43';
	  ELSIF tmp = 'と' THEN
		result:= result || '44';
	  ELSIF tmp = 'ト' THEN
		result:= result || '44';
	  ELSIF tmp = 'ど' THEN
		result:= result || '45';
	  ELSIF tmp = 'ド' THEN
		result:= result || '45';
	  ELSIF tmp = 'な' THEN
		result:= result || '46';
	  ELSIF tmp = 'ナ' THEN
		result:= result || '46';
	  ELSIF tmp = 'に' THEN
		result:= result || '47';
	  ELSIF tmp = 'ニ' THEN
		result:= result || '47';
	  ELSIF tmp = 'ぬ' THEN
		result:= result || '48';
	  ELSIF tmp = 'ヌ' THEN
		result:= result || '48';
	  ELSIF tmp = 'ね' THEN
		result:= result || '49';
	  ELSIF tmp = 'ネ' THEN
		result:= result || '49';
	  ELSIF tmp = 'の' THEN
		result:= result || '50';
	  ELSIF tmp = 'ノ' THEN
		result:= result || '50';
	  ELSIF tmp = 'は' THEN
		result:= result || '51';
	  ELSIF tmp = 'ハ' THEN
		result:= result || '51';
	  ELSIF tmp = 'ば' THEN
		result:= result || '52';
	  ELSIF tmp = 'バ' THEN
		result:= result || '52';
	  ELSIF tmp = 'ぱ' THEN
		result:= result || '53';
	  ELSIF tmp = 'パ' THEN
		result:= result || '53';
	  ELSIF tmp = 'ひ' THEN
		result:= result || '54';
	  ELSIF tmp = 'ヒ' THEN
		result:= result || '54';
	  ELSIF tmp = 'び' THEN
		result:= result || '55';
	  ELSIF tmp = 'ビ' THEN
		result:= result || '55';
	  ELSIF tmp = 'ぴ' THEN
		result:= result || '56';
	  ELSIF tmp = 'ピ' THEN
		result:= result || '56';
	  ELSIF tmp = 'ふ' THEN
		result:= result || '57';
	  ELSIF tmp = 'フ' THEN
		result:= result || '57';
	  ELSIF tmp = 'ぶ' THEN
		result:= result || '58';
	  ELSIF tmp = 'ブ' THEN
		result:= result || '58';
	  ELSIF tmp = 'ぷ' THEN
		result:= result || '59';
	  ELSIF tmp = 'プ' THEN
		result:= result || '59';
	  ELSIF tmp = 'へ' THEN
		result:= result || '60';
	  ELSIF tmp = 'ヘ' THEN
		result:= result || '60';
	  ELSIF tmp = 'べ' THEN
		result:= result || '61';
	  ELSIF tmp = 'ベ' THEN
		result:= result || '61';
	  ELSIF tmp = 'ぺ' THEN
		result:= result || '62';
	  ELSIF tmp = 'ペ' THEN
		result:= result || '62';
	  ELSIF tmp = 'ほ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ホ' THEN
		result:= result || '63';
	  ELSIF tmp = 'ぼ' THEN
		result:= result || '64';
	  ELSIF tmp = 'ボ' THEN
		result:= result || '64';
	  ELSIF tmp = 'ぽ' THEN
		result:= result || '65';
	  ELSIF tmp = 'ポ' THEN
		result:= result || '65';
	  ELSIF tmp = 'ま' THEN
		result:= result || '66';
	  ELSIF tmp = 'マ' THEN
		result:= result || '66';
	  ELSIF tmp = 'み' THEN
		result:= result || '67';
	  ELSIF tmp = 'ミ' THEN
		result:= result || '67';
	  ELSIF tmp = 'む' THEN
		result:= result || '68';
	  ELSIF tmp = 'ム' THEN
		result:= result || '68';
	  ELSIF tmp = 'め' THEN
		result:= result || '69';
	  ELSIF tmp = 'メ' THEN
		result:= result || '69';
	  ELSIF tmp = 'も' THEN
		result:= result || '70';
	  ELSIF tmp = 'モ' THEN
		result:= result || '70';
	  ELSIF tmp = 'ら' THEN
		result:= result || '71';
	  ELSIF tmp = 'ラ' THEN
		result:= result || '71';
	  ELSIF tmp = 'り' THEN
		result:= result || '72';
	  ELSIF tmp = 'リ' THEN
		result:= result || '72';
	  ELSIF tmp = 'る' THEN
		result:= result || '73';
	  ELSIF tmp = 'ル' THEN
		result:= result || '73';
	  ELSIF tmp = 'れ' THEN
		result:= result || '74';
	  ELSIF tmp = 'レ' THEN
		result:= result || '74';
	  ELSIF tmp = 'ろ' THEN
		result:= result || '75';
	  ELSIF tmp = 'ロ' THEN
		result:= result || '75';
	  ELSIF tmp = 'や' THEN
		result:= result || '76';
	  ELSIF tmp = 'ヤ' THEN
		result:= result || '76';
	  ELSIF tmp = 'ゃ' THEN
		result:= result || '76';
	  ELSIF tmp = 'ャ' THEN
		result:= result || '76';
	  ELSIF tmp = 'ゆ' THEN
		result:= result || '77';
	  ELSIF tmp = 'ユ' THEN
		result:= result || '77';
	  ELSIF tmp = 'ゅ' THEN
		result:= result || '77';
	  ELSIF tmp = 'ュ' THEN
		result:= result || '77';
	  ELSIF tmp = 'よ' THEN
		result:= result || '78';
	  ELSIF tmp = 'ヨ' THEN
		result:= result || '78';
	  ELSIF tmp = 'ょ' THEN
		result:= result || '78';
	  ELSIF tmp = 'ョ' THEN
		result:= result || '78';
	  ELSIF tmp = 'わ' THEN
		result:= result || '79';
	  ELSIF tmp = 'ワ' THEN
		result:= result || '79';
	  ELSIF tmp = 'ゐ' THEN
		result:= result || '80';
	  ELSIF tmp = 'ヰ' THEN
		result:= result || '80';
	  ELSIF tmp = 'ゑ' THEN
		result:= result || '81';
	  ELSIF tmp = 'ヱ' THEN
		result:= result || '81';
	  ELSIF tmp = 'を' THEN
		result:= result || '82';
	  ELSIF tmp = 'ヲ' THEN
		result:= result || '82';
	  ELSIF tmp = 'ん' THEN
		result:= result || '83';
	  ELSIF tmp = 'ン' THEN
		result:= result || '83';
	  ELSIF tmp = 'ー' THEN
		result:= result;
	  ELSE
		result:= result || tmp;
	  END IF;
  	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );



-- kau Kanuri 
-- ‘ a b c d e ə f g h i j k l m n ny o p r ɍ s sh t u w y z
CREATE OR REPLACE FUNCTION kau_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
  i		integer := 1;
  tmp		char;
  tmp2		char = SUBSTR( $1, 1, 1 );
  tmp3		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	WHILE i <= length LOOP
		tmp := tmp2;
		tmp2 := SUBSTR( $1, i+1, 1 );
		tmp3 := SUBSTR( $1, i+2, 1 );
  IF tmp = '‘'  THEN
    result:= result || '01';
  ELSIF tmp = '‘'  THEN
    result:= result || '01';
  ELSIF tmp = 'A'  THEN
    result:= result || '02';
  ELSIF tmp = 'a'  THEN
    result:= result || '02';
  ELSIF tmp = 'B' THEN
    result:= result || '03';
  ELSIF tmp = 'b' THEN
    result:= result || '03';
  ELSIF tmp = 'C' THEN
    result:= result || '04';
  ELSIF tmp = 'c' THEN
    result:= result || '04';
  ELSIF tmp = 'D' THEN
    result:= result || '05';
  ELSIF tmp = 'd' THEN
    result:= result || '05';
  ELSIF tmp = 'E' THEN
    result:= result || '06';
  ELSIF tmp = 'e' THEN
    result:= result || '06';
  ELSIF tmp = 'Ə' THEN
    result:= result || '07';
  ELSIF tmp = 'ə' THEN
    result:= result || '07';
  ELSIF tmp = 'Ǝ' THEN
    result:= result || '07';
  ELSIF tmp = 'ǝ' THEN
    result:= result || '07';
  ELSIF tmp = 'F' THEN
	result:= result || '08';
  ELSIF tmp = 'f' THEN
	result:= result || '08';
  ELSIF tmp = 'G' THEN
	result:= result || '09';
  ELSIF tmp = 'g' THEN
	result:= result || '09';
  ELSIF tmp = 'H' THEN
    result:= result || '10';
  ELSIF tmp = 'h' THEN
    result:= result || '10';
  ELSIF tmp = 'I' THEN
    result:= result || '11';
  ELSIF tmp = 'i' THEN
    result:= result || '11';
  ELSIF tmp = 'J' THEN
    result:= result || '12';
  ELSIF tmp = 'j' THEN
    result:= result || '12';
  ELSIF tmp = 'K' THEN
	result:= result || '13';
  ELSIF tmp = 'k' THEN
	result:= result || '13';
  ELSIF tmp = 'L' THEN
    result:= result || '14';
  ELSIF tmp = 'l' THEN
    result:= result || '14';
  ELSIF tmp = 'M' THEN
    result:= result || '15';
  ELSIF tmp = 'm' THEN
    result:= result || '15';
  ELSIF tmp = 'N' THEN
	IF tmp2 = 'Y' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '17';
	ELSIF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '17';
	ELSE
		result:= result || '16';
	END IF;
  ELSIF tmp = 'n' THEN
	IF tmp2 = 'y' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '17';
	ELSE
		result:= result || '16';
	END IF;
  ELSIF tmp = 'O' THEN
    result:= result || '18';
  ELSIF tmp = 'o' THEN
    result:= result || '18';
  ELSIF tmp = 'p' THEN
    result:= result || '19';
  ELSIF tmp = 'P' THEN
    result:= result || '19';
  ELSIF tmp = 'R' THEN
    result:= result || '20';
  ELSIF tmp = 'r' THEN
    result:= result || '20';
  ELSIF tmp = 'Ɍ' THEN
    result:= result || '21';
  ELSIF tmp = 'ɍ' THEN
    result:= result || '21';
  ELSIF tmp = 'S' THEN
	IF tmp2 = 'H' THEN
		tmp2 := tmp3;
		i := i+1;
		result:= result || '23';
	ELSIF tmp2 = 'h' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '23';
	ELSE
		result:= result || '22';
	END IF;
  ELSIF tmp = 's' THEN
	IF tmp2 = 'h' THEN
		tmp2 := tmp3;
		i := i+1;
	    result:= result || '23';
	ELSE
		result:= result || '22';
	END IF;
  ELSIF tmp = 'T' THEN
	result:= result || '24';
  ELSIF tmp = 't' THEN
	result:= result || '24';
  ELSIF tmp = 'U' THEN
    result:= result || '25';
  ELSIF tmp = 'u' THEN
    result:= result || '25';
  ELSIF tmp = 'W' THEN
    result:= result || '26';
  ELSIF tmp = 'w' THEN
    result:= result || '26';
  ELSIF tmp = 'Y' THEN
    result:= result || '27';
  ELSIF tmp = 'y' THEN
    result:= result || '27';
  ELSIF tmp = 'Z' THEN
    result:= result || '28';
  ELSIF tmp = 'z' THEN
    result:= result || '28';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
  i:=i+1;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- kor Korean
-- 
-- for the moment do nothing
-- todo: maybe sort by unicode code
CREATE OR REPLACE FUNCTION kor_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );



-- msa Malay
-- a b c d e f g h i j k l m n o p q r s t u v w x y z
-- for the moment do nothing; probably ok as is
CREATE OR REPLACE FUNCTION msa_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- nru Yongning Na
-- [ɑ, ɑ̃, æ, æ̃, b, ɕ, d, dz, dʑ, ɖ, ɖʐ, e, ɤ, ɤ̃, ə, f, g, ɣ, h, i, ĩ, j, jæ, jɤ, je, jo, ʝ, k, kʰ, l, ɭ, ɬ, m, n, ɳ, ɲ, ŋ, o, õ, p, pʰ, q, qʰ, ɻ, ɻ̍, ɻ ̍̃, ʁ, s, ʂ, t, tʰ, tɕ, tɕʰ, ts, tsʰ, ʈ, ʈʰ, ʈʂ, ʈʂʰ, u, ɯ, v̩, [ṽ̩, ṽ̩], w, wɑ, wæ, wɤ, wo, wɤ̃, w̃, w̃æ, z, ʐ, ʑ, x, [˥, ˥₁, ˥₂, ˥₃, ˧, ˧₁, ˧₂, ˧₃, ˩, ˩₁, ˩₂, ˩₃, ˧˥, ˧˥₁, ˧˥₂, ˧˥₃, ˩˥, ˩˥₁, ˩˥₂, ˩˥₃, ˩˧, ˩˧₁, ˩˧₂, ˩˧₃]]
-- 10 ɑ, ɑ̃, æ, æ̃, b, ɕ, d, dz, dʑ, ɖ,
-- 20 ɖʐ, e, ɤ, ɤ̃, ə, f, g, ɣ, h, i,
-- 30 ĩ, j, jæ, jɤ, je, jo, ʝ, k, kʰ, l,
-- 40 ɭ, ɬ, m, n, ɳ, ɲ, ŋ, o, õ, p,
-- 50 pʰ, q, qʰ, ɻ, ɻ̍, ɻ ̍̃, ʁ, s, ʂ, t,
-- 60 tʰ, tɕ, tɕʰ, ts, tsʰ, ʈ, ʈʰ, ʈʂ, ʈʂʰ, u,
-- 70 ɯ, v̩, ṽ̩, w, wɑ, wæ, wɤ, wo, wɤ̃, w̃,
-- 80 w̃æ, z, ʐ, ʑ, x

-- ˥ ˧ ˩ ˧˥ ˩˥ ˩˧
-- 01 ˥, ˥₁, ˥₂, ˥₃,
-- 02 ˧, ˧₁, ˧₂, ˧₃, 
-- 03 ˩, ˩₁, ˩₂, ˩₃, 
-- 04 ˧˥, ˧˥₁, ˧˥₂, ˧˥₃, 
-- 05 ˩˥, ˩˥₁, ˩˥₂, ˩˥₃, 
-- 06 ˩˧, ˩˧₁, ˩˧₂, ˩˧₃



CREATE OR REPLACE FUNCTION nru_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  i		integer := 1;
  tmp		char;
  tmp2		char = SUBSTR( $1, 1, 1 );
  tmp3		char = SUBSTR( $1, 2, 1 );
  tmp4		char;
  result	varchar := '';
  length	integer;
 BEGIN
 	length := char_length($1);
  	WHILE i <= length LOOP
		tmp := tmp2;
		tmp2 := tmp3;
		tmp3 := SUBSTR( $1, i+2, 1 );
		tmp4 := SUBSTR( $1, i+3, 1 );
  IF tmp = 'ɑ'  THEN
	IF tmp2 = '̃' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '11';
	ELSE
		result:= result || '10';
	END IF;
  ELSIF tmp = 'æ'  THEN
	IF tmp2 = '̃' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '13';
	ELSE
		result:= result || '12';
	END IF;
  ELSIF tmp = 'b' THEN
    result:= result || '14';
  ELSIF tmp = 'ɕ' THEN
    result:= result || '15';
  ELSIF tmp = 'd'  THEN
	IF tmp2 = 'z' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '17';
	ELSIF tmp2 = 'ʑ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '18';
	ELSE
		result:= result || '16';
	END IF;
  ELSIF tmp = 'ɖ'  THEN
	IF tmp2 = 'ʐ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '20';
	ELSE
		result:= result || '19';
	END IF;
  ELSIF tmp = 'e' THEN
    result:= result || '21';
  ELSIF tmp = 'ɤ'  THEN
	IF tmp2 = '̃' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '23';
	ELSE
		result:= result || '22';
	END IF;
  ELSIF tmp = 'ə' THEN
    result:= result || '24';
  ELSIF tmp = 'f' THEN
    result:= result || '25';
  ELSIF tmp = 'g' THEN
    result:= result || '26';
  ELSIF tmp = 'ɣ' THEN
    result:= result || '27';
  ELSIF tmp = 'h' THEN
    result:= result || '28';
  ELSIF tmp = 'i' THEN
    result:= result || '29';
  ELSIF tmp = 'ĩ' THEN
    result:= result || '30';
  ELSIF tmp = 'j'  THEN
	IF tmp2 = 'æ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '32';
	ELSIF tmp2 = 'ɤ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '33';
	ELSIF tmp2 = 'e' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '34';
	ELSIF tmp2 = 'o' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '35';
	ELSE
		result:= result || '31';
	END IF;
  ELSIF tmp = 'ʝ' THEN
    result:= result || '36';
  ELSIF tmp = 'k' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '38';
	ELSE
		result:= result || '37';
	END IF;
  ELSIF tmp = 'l' THEN
    result:= result || '39';
  ELSIF tmp = 'ɭ' THEN
    result:= result || '40';
  ELSIF tmp = 'ɬ' THEN
    result:= result || '41';
  ELSIF tmp = 'm' THEN
    result:= result || '42';
  ELSIF tmp = 'n' THEN
    result:= result || '43';
  ELSIF tmp = 'ɳ' THEN
    result:= result || '44';
  ELSIF tmp = 'ɲ' THEN
    result:= result || '45';
  ELSIF tmp = 'ŋ' THEN
    result:= result || '46';
  ELSIF tmp = 'o' THEN
    result:= result || '47';
  ELSIF tmp = 'õ' THEN
    result:= result || '48';
  ELSIF tmp = 'p' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '50';
	ELSE
		result:= result || '49';
	END IF;
  ELSIF tmp = 'q' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '52';
	ELSE
		result:= result || '51';
	END IF;
  ELSIF tmp = 'ɻ' THEN
	IF tmp2 = '̍' THEN
		IF tmp3 = '̃' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '55';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '54';
		END IF;
	ELSE
		result:= result || '53';
	END IF;
  ELSIF tmp = 'ʁ' THEN
    result:= result || '56';
  ELSIF tmp = 's' THEN
    result:= result || '57';
  ELSIF tmp = 'ʂ' THEN
    result:= result || '58';
  ELSIF tmp = 't' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '60';
	ELSIF tmp2 = 'ɕ' THEN
		IF tmp3 = 'ʰ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '62';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '61';
		END IF;
	ELSIF tmp2 = 's' THEN
		IF tmp3 = 'ʰ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '64';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '63';
		END IF;
	ELSE
		result:= result || '59';
	END IF;
  ELSIF tmp = 'ʈ' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '66';
	ELSIF tmp2 = 'ʂ' THEN
		IF tmp3 = 'ʰ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '68';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '67';
		END IF;
	ELSE
		result:= result || '65';
	END IF;
  ELSIF tmp = 'u' THEN
    result:= result || '69';
  ELSIF tmp = 'ɯ' THEN
    result:= result || '70';
  ELSIF tmp = 'v' THEN
    result:= result || '71';
  ELSIF tmp = 'ṽ' THEN
    result:= result || '72';
  ELSIF tmp = '̩' THEN
    result:= result;
  ELSIF tmp = 'w' THEN
	IF tmp2 = 'ɑ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '74';
	ELSIF tmp2 = 'æ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '75';
	ELSIF tmp2 = 'o' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '77';
	ELSIF tmp2 = 'ɤ' THEN
		IF tmp3 = '̃' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '78';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '76';
		END IF;
	ELSIF tmp2 = '̃' THEN
		IF tmp3 = 'æ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '80';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '79';
		END IF;
	ELSE
		result:= result || '73';
	END IF;
  ELSIF tmp = 'z' THEN
    result:= result || '81';
  ELSIF tmp = 'ʐ' THEN
    result:= result || '82';
  ELSIF tmp = 'ʑ' THEN
    result:= result || '83';
  ELSIF tmp = 'x' THEN
    result:= result || '84';
	
-- les tons
  ELSIF tmp = '˥' THEN
    result:= result;
  ELSIF tmp = '˧' THEN
    result:= result;
  ELSIF tmp = '˩' THEN
    result:= result;
  ELSIF tmp = '₁' THEN
    result:= result;
  ELSIF tmp = '₂' THEN
    result:= result;
  ELSIF tmp = '₃' THEN
    result:= result;
  ELSIF tmp = 'α' THEN
    result:= result;
  ELSIF tmp = 'β' THEN
    result:= result;
  ELSIF tmp = 'γ' THEN
    result:= result;

-- fin	
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
  i:=i+1;
  END LOOP;
  result:= result || '00';
  i:=1;
  tmp2 = SUBSTR( $1, 1, 1 );
  tmp3 = SUBSTR( $1, 2, 1 );
  WHILE i <= length LOOP
		tmp := tmp2;
		tmp2 := tmp3;
		tmp3 := SUBSTR( $1, i+2, 1 );
		tmp4 := SUBSTR( $1, i+3, 1 );
	IF tmp = 'ɑ'  THEN
	IF tmp2 = '̃' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '11';
	ELSE
		result:= result || '10';
	END IF;
  ELSIF tmp = 'æ'  THEN
	IF tmp2 = '̃' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '13';
	ELSE
		result:= result || '12';
	END IF;
  ELSIF tmp = 'b' THEN
    result:= result || '14';
  ELSIF tmp = 'ɕ' THEN
    result:= result || '15';
  ELSIF tmp = 'd'  THEN
	IF tmp2 = 'z' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '17';
	ELSIF tmp2 = 'ʑ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '18';
	ELSE
		result:= result || '16';
	END IF;
  ELSIF tmp = 'ɖ'  THEN
	IF tmp2 = 'ʐ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '20';
	ELSE
		result:= result || '19';
	END IF;
  ELSIF tmp = 'e' THEN
    result:= result || '21';
  ELSIF tmp = 'ɤ'  THEN
	IF tmp2 = '̃' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '23';
	ELSE
		result:= result || '22';
	END IF;
  ELSIF tmp = 'ə' THEN
    result:= result || '24';
  ELSIF tmp = 'f' THEN
    result:= result || '25';
  ELSIF tmp = 'g' THEN
    result:= result || '26';
  ELSIF tmp = 'ɣ' THEN
    result:= result || '27';
  ELSIF tmp = 'h' THEN
    result:= result || '28';
  ELSIF tmp = 'i' THEN
    result:= result || '29';
  ELSIF tmp = 'ĩ' THEN
    result:= result || '30';
  ELSIF tmp = 'j'  THEN
	IF tmp2 = 'æ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '32';
	ELSIF tmp2 = 'ɤ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '33';
	ELSIF tmp2 = 'e' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '34';
	ELSIF tmp2 = 'o' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '35';
	ELSE
		result:= result || '31';
	END IF;
  ELSIF tmp = 'ʝ' THEN
    result:= result || '36';
  ELSIF tmp = 'k' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '38';
	ELSE
		result:= result || '37';
	END IF;
  ELSIF tmp = 'l' THEN
    result:= result || '39';
  ELSIF tmp = 'ɭ' THEN
    result:= result || '40';
  ELSIF tmp = 'ɬ' THEN
    result:= result || '41';
  ELSIF tmp = 'm' THEN
    result:= result || '42';
  ELSIF tmp = 'n' THEN
    result:= result || '43';
  ELSIF tmp = 'ɳ' THEN
    result:= result || '44';
  ELSIF tmp = 'ɲ' THEN
    result:= result || '45';
  ELSIF tmp = 'ŋ' THEN
    result:= result || '46';
  ELSIF tmp = 'o' THEN
    result:= result || '47';
  ELSIF tmp = 'õ' THEN
    result:= result || '48';
  ELSIF tmp = 'p' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '50';
	ELSE
		result:= result || '49';
	END IF;
  ELSIF tmp = 'q' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '52';
	ELSE
		result:= result || '51';
	END IF;
  ELSIF tmp = 'ɻ' THEN
	IF tmp2 = '̍' THEN
		IF tmp3 = '̃' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '55';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '54';
		END IF;
	ELSE
		result:= result || '53';
	END IF;
  ELSIF tmp = 'ʁ' THEN
    result:= result || '56';
  ELSIF tmp = 's' THEN
    result:= result || '57';
  ELSIF tmp = 'ʂ' THEN
    result:= result || '58';
  ELSIF tmp = 't' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '60';
	ELSIF tmp2 = 'ɕ' THEN
		IF tmp3 = 'ʰ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '62';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '61';
		END IF;
	ELSIF tmp2 = 's' THEN
		IF tmp3 = 'ʰ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '64';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '63';
		END IF;
	ELSE
		result:= result || '59';
	END IF;
  ELSIF tmp = 'ʈ' THEN
	IF tmp2 = 'ʰ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '66';
	ELSIF tmp2 = 'ʂ' THEN
		IF tmp3 = 'ʰ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '68';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '67';
		END IF;
	ELSE
		result:= result || '65';
	END IF;
  ELSIF tmp = 'u' THEN
    result:= result || '69';
  ELSIF tmp = 'ɯ' THEN
    result:= result || '70';
  ELSIF tmp = 'v' THEN
    result:= result || '71';
  ELSIF tmp = 'ṽ' THEN
    result:= result || '72';
  ELSIF tmp = '̩' THEN
    result:= result;
  ELSIF tmp = 'w' THEN
	IF tmp2 = 'ɑ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '74';
	ELSIF tmp2 = 'æ' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '75';
	ELSIF tmp2 = 'o' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '77';
	ELSIF tmp2 = 'ɤ' THEN
		IF tmp3 = '̃' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '78';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '76';
		END IF;
	ELSIF tmp2 = '̃' THEN
		IF tmp3 = 'æ' THEN
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+2;
			result:= result || '80';
		ELSE
			tmp2 := tmp3;
			tmp3 := tmp4;
			i := i+1;
			result:= result || '79';
		END IF;
	ELSE
		result:= result || '73';
	END IF;
  ELSIF tmp = 'z' THEN
    result:= result || '81';
  ELSIF tmp = 'ʐ' THEN
    result:= result || '82';
  ELSIF tmp = 'ʑ' THEN
    result:= result || '83';
  ELSIF tmp = 'x' THEN
    result:= result || '84';
	
-- les tons
  ELSIF tmp = '˥' THEN
	result:= result || '01';
  ELSIF tmp = '˧' THEN
	IF tmp2 = '˥' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '04';
	ELSE
		result:= result || '02';
	END IF;
  ELSIF tmp = '˩' THEN
	IF tmp2 = '˥' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '05';
	ELSIF tmp2 = '˧' THEN
		tmp2 := tmp3;
		tmp3 := tmp4;
		i := i+1;
		result:= result || '06';
	ELSE
		result:= result || '03';
	END IF;
  ELSIF tmp = '₁' THEN
    result:= result;
  ELSIF tmp = '₂' THEN
    result:= result;
  ELSIF tmp = '₃' THEN
    result:= result;
  ELSIF tmp = 'α' THEN
    result:= result;
  ELSIF tmp = 'β' THEN
    result:= result;
  ELSIF tmp = 'γ' THEN
    result:= result;

-- fin	
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
  i:=i+1;
	END LOOP;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );




-- slo Slovak
--
-- for the moment do nothing
-- todo: maybe slovak sort 
CREATE OR REPLACE FUNCTION slo_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$  LANGUAGE 'plpgsql' WITH ( ISCACHABLE );

-- slv Slovene
--
-- A B C Č D E F G H I J K L M N O P R S Š š T U V Z Ž ž
CREATE OR REPLACE FUNCTION slv_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
  tmp     char;
  result  varchar := '';
  length  integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'B' THEN
    result:= result || '02';
  ELSIF tmp = 'b' THEN
    result:= result || '02';
  ELSIF tmp = 'C' THEN
    result:= result || '03';
  ELSIF tmp = 'c' THEN
    result:= result || '03';
  ELSIF tmp = 'Č' THEN
    result:= result || '04';
  ELSIF tmp = 'č' THEN
    result:= result || '04';
  ELSIF tmp = 'D' THEN
    result:= result || '05';
  ELSIF tmp = 'd' THEN
    result:= result || '05';
  ELSIF tmp = 'E' THEN
    result:= result || '06';
  ELSIF tmp = 'e' THEN
    result:= result || '06';
  ELSIF tmp = 'F' THEN
    result:= result || '07';
  ELSIF tmp = 'f' THEN
    result:= result || '07';
  ELSIF tmp = 'G' THEN
    result:= result || '08';
  ELSIF tmp = 'g' THEN
    result:= result || '08';
  ELSIF tmp = 'H' THEN
    result:= result || '09';
  ELSIF tmp = 'h' THEN
    result:= result || '09';
  ELSIF tmp = 'I' THEN
    result:= result || '10';
  ELSIF tmp = 'i' THEN
    result:= result || '10';
  ELSIF tmp = 'J' THEN
    result:= result || '11';
  ELSIF tmp = 'j' THEN
    result:= result || '11';
  ELSIF tmp = 'K' THEN
    result:= result || '12';
  ELSIF tmp = 'k' THEN
    result:= result || '12';
  ELSIF tmp = 'L' THEN
    result:= result || '13';
  ELSIF tmp = 'l' THEN
    result:= result || '13';
  ELSIF tmp = 'M' THEN
    result:= result || '14';
  ELSIF tmp = 'm' THEN
    result:= result || '14';
  ELSIF tmp = 'N' THEN
    result:= result || '15';
  ELSIF tmp = 'n' THEN
    result:= result || '15';
  ELSIF tmp = 'O' THEN
    result:= result || '16';
  ELSIF tmp = 'o' THEN
    result:= result || '16';
  ELSIF tmp = 'p' THEN
    result:= result || '17';
  ELSIF tmp = 'P' THEN
    result:= result || '17';
  ELSIF tmp = 'R' THEN
    result:= result || '19';
  ELSIF tmp = 'r' THEN
    result:= result || '19';
  ELSIF tmp = 'S' THEN
    result:= result || '20';
  ELSIF tmp = 's' THEN
    result:= result || '20';
  ELSIF tmp = 'Š' THEN
    result:= result || '21';
  ELSIF tmp = 'š' THEN
    result:= result || '21';
  ELSIF tmp = 'T' THEN
    result:= result || '22';
  ELSIF tmp = 't' THEN
    result:= result || '22';
  ELSIF tmp = 'U' THEN
    result:= result || '23';
  ELSIF tmp = 'u' THEN
    result:= result || '23';
  ELSIF tmp = 'V' THEN
    result:= result || '24';
  ELSIF tmp = 'v' THEN
    result:= result || '24';
  ELSIF tmp = 'Z' THEN
    result:= result || '28';
  ELSIF tmp = 'z' THEN
    result:= result || '28';
  ELSIF tmp = 'Ž' THEN
    result:= result || '29';
  ELSIF tmp = 'ž' THEN
    result:= result || '29';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
   END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- tha Thai
-- 
-- for the moment do nothing
-- todo: maybe sort by unicode code
CREATE OR REPLACE FUNCTION tha_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );

-- tmh Tamajaq
-- a ǎ b c d ḍ e ə f g ğ h i j ǰ ɣ k x l ḷ m n ŋ o q r s ṣ š t ṭ u w y z ẓ
CREATE OR REPLACE FUNCTION tmh_sort( varchar )
 RETURNS varchar AS $PROC$

 DECLARE
  tmp     char;
  result  varchar := '';
  length  integer;
 BEGIN
 	length := char_length($1);
  	FOR i IN 1.. length LOOP
  		tmp := SUBSTR( $1, i, 1 );
  IF tmp = 'A'  THEN
    result:= result || '01';
  ELSIF tmp = 'a'  THEN
    result:= result || '01';
  ELSIF tmp = 'Ǎ'  THEN
    result:= result || '02';
  ELSIF tmp = 'ǎ'  THEN
    result:= result || '02';
  ELSIF tmp = 'B' THEN
    result:= result || '03';
  ELSIF tmp = 'b' THEN
    result:= result || '03';
  ELSIF tmp = 'C' THEN
    result:= result || '04';
  ELSIF tmp = 'c' THEN
    result:= result || '04';
  ELSIF tmp = 'D' THEN
    result:= result || '05';
  ELSIF tmp = 'd' THEN
    result:= result || '05';
  ELSIF tmp = 'Ḍ' THEN
    result:= result || '06';
  ELSIF tmp = 'ḍ' THEN
    result:= result || '06';
  ELSIF tmp = 'E' THEN
    result:= result || '07';
  ELSIF tmp = 'e' THEN
    result:= result || '07';
  ELSIF tmp = 'Ə' THEN
    result:= result || '08';
  ELSIF tmp = 'ə' THEN
    result:= result || '08';
  ELSIF tmp = 'Ǝ' THEN
    result:= result || '08';
  ELSIF tmp = 'ǝ' THEN
    result:= result || '08';
  ELSIF tmp = 'F' THEN
    result:= result || '09';
  ELSIF tmp = 'f' THEN
    result:= result || '09'; 
  ELSIF tmp = 'G' THEN
    result:= result || '10';
  ELSIF tmp = 'g' THEN
    result:= result || '10';
  ELSIF tmp = 'Ğ' THEN
    result:= result || '11';
  ELSIF tmp = 'ğ' THEN
    result:= result || '11';
  ELSIF tmp = 'H' THEN
    result:= result || '12';
  ELSIF tmp = 'h' THEN
    result:= result || '12';
  ELSIF tmp = 'I' THEN
    result:= result || '13';
  ELSIF tmp = 'i' THEN
    result:= result || '13';
  ELSIF tmp = 'J' THEN
    result:= result || '14';
  ELSIF tmp = 'j' THEN
    result:= result || '14';
  ELSIF tmp = 'J' THEN
    result:= result || '15';
  ELSIF tmp = 'ǰ' THEN
    result:= result || '15'; 
  ELSIF tmp = 'Ɣ' THEN
    result:= result || '16';
  ELSIF tmp = 'ɣ' THEN
    result:= result || '16'; 
  ELSIF tmp = 'K' THEN
    result:= result || '17';
  ELSIF tmp = 'k' THEN
    result:= result || '17';
  ELSIF tmp = 'X' THEN
    result:= result || '18';
  ELSIF tmp = 'x' THEN
    result:= result || '18';
  ELSIF tmp = 'L' THEN
    result:= result || '19';
  ELSIF tmp = 'l' THEN
    result:= result || '19';
  ELSIF tmp = 'Ḷ' THEN
    result:= result || '20';
  ELSIF tmp = 'ḷ' THEN
    result:= result || '20';
  ELSIF tmp = 'M' THEN
    result:= result || '21';
  ELSIF tmp = 'm' THEN
    result:= result || '21';
  ELSIF tmp = 'N' THEN
    result:= result || '22';
  ELSIF tmp = 'n' THEN
    result:= result || '22';
  ELSIF tmp = 'Ŋ' THEN
    result:= result || '23';
  ELSIF tmp = 'ŋ' THEN
    result:= result || '23';
  ELSIF tmp = 'O' THEN
    result:= result || '24';
  ELSIF tmp = 'o' THEN
    result:= result || '24';
  ELSIF tmp = 'p' THEN
    result:= result || '25';
  ELSIF tmp = 'P' THEN
    result:= result || '25';
  ELSIF tmp = 'Q' THEN
    result:= result || '26';
  ELSIF tmp = 'q' THEN
    result:= result || '26';
  ELSIF tmp = 'R' THEN
    result:= result || '27';
  ELSIF tmp = 'r' THEN
    result:= result || '27';
  ELSIF tmp = 'S' THEN
    result:= result || '28';
  ELSIF tmp = 's' THEN
    result:= result || '28';
  ELSIF tmp = 'Ṣ' THEN
    result:= result || '29';
  ELSIF tmp = 'ṣ' THEN
    result:= result || '29';
  ELSIF tmp = 'Š' THEN
    result:= result || '30';
  ELSIF tmp = 'š' THEN
    result:= result || '30';
  ELSIF tmp = 'T' THEN
    result:= result || '31';
  ELSIF tmp = 't' THEN
    result:= result || '31';
  ELSIF tmp = 'Ṭ' THEN
    result:= result || '32';
  ELSIF tmp = 'ṭ' THEN
    result:= result || '32';
  ELSIF tmp = 'U' THEN
    result:= result || '33';
  ELSIF tmp = 'u' THEN
    result:= result || '33';
  ELSIF tmp = 'V' THEN
    result:= result || '34';
  ELSIF tmp = 'v' THEN
    result:= result || '34';
  ELSIF tmp = 'W' THEN
    result:= result || '35';
  ELSIF tmp = 'w' THEN
    result:= result || '35';
  ELSIF tmp = 'Y' THEN
    result:= result || '36';
  ELSIF tmp = 'y' THEN
    result:= result || '36';
  ELSIF tmp = 'Z' THEN
    result:= result || '37';
  ELSIF tmp = 'z' THEN
    result:= result || '37';
  ELSIF tmp = 'Ẓ' THEN
    result:= result || '38';
  ELSIF tmp = 'ẓ' THEN
    result:= result || '38';
  ELSIF tmp = '̌' THEN
    result:= result || '';
  ELSIF tmp = ' ' THEN
    result:= result;
  ELSIF tmp = '-' THEN
    result:= result;
  ELSIF tmp = '+' THEN
    result:= result;
  ELSIF tmp = '#' THEN
    result:= result;
  ELSE
    result:= result || '99';
  END IF;
	END LOOP;
  return( result );
   END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );


-- vie Vietnamese
-- a b c d e f g h i j k l m n o p q r s t u v w x y z
-- for the moment do nothing
-- todo: umaybe sort by unicode code or suppress diacritics ...
CREATE OR REPLACE FUNCTION vie_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );



-- zho Chinese
-- for the moment do nothing
-- todo: sort by unicode code or reading...
CREATE OR REPLACE FUNCTION zho_sort( varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  tmp		char;
  result	varchar := '';
  length	integer;
 BEGIN
  return( $1 );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );

-- multilingual sort
CREATE OR REPLACE FUNCTION multilingual_sort( varchar,varchar ) 
 RETURNS varchar AS $PROC$ 

 DECLARE
  lang	varchar;
  word	varchar;
  result	varchar := '';
 BEGIN
  lang := $1;
  word := $2;
   IF lang = 'bam' THEN
    result:= bam_sort(word);
  ELSIF lang = 'deu' THEN
    result:= deu_sort(word);
  ELSIF lang = 'dje' THEN
    result:= dje_sort(word);
  ELSIF lang = 'eng' THEN
    result:= eng_sort(word);
  ELSIF lang = 'est' THEN
    result:= est_sort(word);
  ELSIF lang = 'fra' THEN
    result:= fra_sort(word);
  ELSIF lang = 'ful' THEN
    result:= ful_sort(word);
  ELSIF lang = 'hau' THEN
    result:= hau_sort(word);
  ELSIF lang = 'ita' THEN
    result:= ita_sort(word);
  ELSIF lang = 'jpn' THEN
    result:= jpn_sort(word);
  ELSIF lang = 'kau' THEN
    result:= kau_sort(word);
  ELSIF lang = 'kor' THEN
    result:= kor_sort(word);
 ELSIF lang = 'msa' THEN
    result:= msa_sort(word);
 ELSIF lang = 'slv' THEN
    result:= slv_sort(word);
 ELSIF lang = 'tha' THEN
    result:= tha_sort(word);
 ELSIF lang = 'tmh' THEN
    result:= tmh_sort(word);
 ELSIF lang = 'vie' THEN
    result:= vie_sort(word);
 ELSIF lang = 'zho' THEN
    result:= zho_sort(word);
 ELSE 
    result:= word;
 END IF;
  result := lang || result;
  return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );

CREATE OR REPLACE FUNCTION complete_numbers( varchar ) 
 RETURNS varchar AS $PROC$

 DECLARE
  first	char;
  zeros	varchar := '000000000000000000000000';
  result	varchar := '';
  lvar		integer;
  lempty	integer;
 BEGIN
 	lvar := char_length($1);
 	first := SUBSTR( $1, 1, 1 );
 	lempty := char_length(zeros);
	IF first >= '0' AND first <= '9' THEN
		result := $1;
		IF lvar < lempty THEN
    			result:= SUBSTR( zeros, 1, lempty-lvar ) || result;
    		END IF;
  	ELSE
    		result:= '';
  	END IF; 
  	return( result );
 END;
$PROC$ LANGUAGE 'plpgsql' WITH ( ISCACHABLE );