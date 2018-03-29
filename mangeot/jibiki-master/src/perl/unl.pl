#!/usr/bin/env perl -w

# see http://papillon.imag.fr/papillon/Api.po for usage info of the Jibiki REST API.

#my $SERVER = 'http://papillon.imag.fr/papillon';
my $SERVER = 'http://aximag.fr:8999/pivax';

use LWP::Simple;
use List::MoreUtils qw/uniq/;

if (!$ARGV[0]) {print "usage: $0 \"the word (in UTF-8 !)\" [source-lang=fra|eng|...] [target-lang=fra|eng|...]\nlang = ISO-639-2/T code";exit};

my $lang = $ARGV[1] || 'fra';
my $trg = $ARGV[2] || 'unl';
$TRG = uc($trg);
$LANG = uc($lang);

&htmlentities('chaine');

foreach my $mot (split(/ /,$ARGV[0])) {
	$lexie = 	get($SERVER.'/api/CommonUNLDict/'.$lang.'/cdm-headword/' . $mot . '/cdm-entry-id/?strategy=EQUAL');
	foreach my $matchlexie (uniq sort ($lexie =~ /<key value="cdm-entry-id">([^<]+)<\/key>/g)) {
		$matchlexie = &htmlentities($matchlexie);
		print 'matchlexie;',$lang,':',$matchlexie,"\n";
		$axeme = get($SERVER.'/api/CommonUNLDict/'.$LANG.'/cdm-translation-ref/' . $matchlexie . '/cdm-entry-id/?strategy=EQUAL');
		foreach my $matchaxeme (uniq sort ($axeme =~ /<key value="cdm-entry-id">([^<]+)<\/key>/g)) {
			$matchaxeme = &htmlentities($matchaxeme);
			print 'matchaxeme;',$LANG,':',$matchaxeme,"\n";
			$axie = get($SERVER.'/api/CommonUNLDict/axi/cdm-translation-ref/' . $matchaxeme . '/cdm-translation-ref/?strategy=EQUAL');
			foreach my $matchaxie (uniq sort ($axie =~ /<key lang="$TRG" value="cdm-translation-ref">([^<]+)<\/key>/g)) {
				$matchaxie = &htmlentities($matchaxie);
				print 'matchaxie;axi:',$matchaxie,"\n";
				#print $SERVER.'/api/CommonUNLDict/'.$trg.'/cdm-entry-id/' . $matchaxie . '/cdm-translation-ref/?strategy=EQUAL';
				$axeme2 = get($SERVER.'/api/CommonUNLDict/'.$TRG.'/cdm-entry-id/' . $matchaxie . '/cdm-translation-ref/?strategy=EQUAL');
				foreach my $matchaxeme2 (uniq sort ($axeme2 =~ /<key lang="$trg" value="cdm-translation-ref">([^<]+)<\/key>/g)) {
					$matchaxeme2 = &htmlentities($matchaxeme2);
					print 'matchaxeme2;',uc($trg),':',$matchaxeme2,"\n";
					if ($trg eq 'unl') {
						$lexie2 = get($SERVER.'/api/CommonUNLDict/'.$trg.'/cdm-entry-id/' . $matchaxeme2 . '/cdm-headword-variant/?strategy=EQUAL');
						foreach my $matchlexie2 (uniq sort ($lexie2 =~ /<key value="cdm-headword-variant">([^<]+)<\/key>/g)) {
							$matchlexie2 = &htmlentities($matchlexie2);
							print 'matchlexie2;',$trg,':',$matchlexie2, "\n";
						}
					}
					else {
						$lexie2 = get($SERVER.'/api/CommonUNLDict/'.$trg.'/cdm-entry-id/' . $matchaxeme2 . '/cdm-headword/?strategy=EQUAL');
						foreach my $matchlexie2 (uniq sort ($lexie2 =~ /<key value="cdm-headword">([^<]+)<\/key>/g)) {
							$matchlexie2 = &htmlentities($matchlexie2);
							print 'matchlexie2;',$trg,':',$matchlexie2, "\n";
						}
					}
				}
			}
		}
	}
}

sub htmlentities  {
	my $string = pop(@_);
	$string =~ s/&gt;/>/g;
	$string =~ s/&lt;/</g;
	$string =~ s/&apos;/'/g;
	return $string;
}