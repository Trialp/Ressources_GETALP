#!/usr/bin/env perl -w

# see http://papillon.imag.fr/papillon/Api.po for usage info of the Jibiki REST API.

use LWP::Simple;

if (!$ARGV[0]) {print "usage: $0 \"la phrase à traduire (en UTF-8 !)\" [lang=eng|msa|vie|tha]\n";exit};

$ARGV[0] =~ s/'/e /g;
$ARGV[0] =~ s/[\.\?\!]//g;
my $lang = $ARGV[1] || 'eng';

foreach my $mot (split(/ /,$ARGV[0])) {
	my $lemme = get('http://papillon.imag.fr/papillon/api/Morphalou/fra/cdm-headword-variant/' . $mot . '/cdm-headword/?strategy=EQUAL&count=1');
	$lemme = $lemme =~ /<key value="cdm-headword">([^<]+)<\/key>/ ? $1 : $mot;
	my $trad = get('http://papillon.imag.fr/papillon/api/FeV/fra/cdm-headword/' . $lemme . '/cdm-translation/?strategy=EQUAL&count=1');
	$trad = $trad =~ /<key lang="$lang" value="cdm-translation">([^<]+)<\/key>/ ? $1 : $lemme;
	print $trad,' ';
}
print "\n";