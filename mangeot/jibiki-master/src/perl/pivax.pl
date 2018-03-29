#!/usr/bin/env perl -w

# see http://papillon.imag.fr/papillon/Api.po for usage info of the Jibiki REST API.
#my $SERVER = 'http://papillon.imag.fr/papillon';
#my $SERVER = 'http://localhost:8999/papillon';
my $SERVER = 'http://aximag.fr:8999/pivax';

use LWP::Simple;

if (!$ARGV[0]) {print "usage: $0 \"the word (in UTF-8 !)\" [lang=fra|eng|...]\nlang = ISO-639-2/T code";exit};

$ARGV[0] =~ s/'/e /g;
$ARGV[0] =~ s/[\.\?\!]//g;
my $lang = $ARGV[1] || 'fra';

foreach my $mot (split(/ /,$ARGV[0])) {
	$lexie = get($SERVER . '/api/CommonUNLDict/'.$lang.'/pivax/' . $mot . '/?strategy=EQUAL');
	print $lexie;
}