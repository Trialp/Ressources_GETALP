#!/usr/bin/env perl -w

use strict;
use LWP::Simple;
use LWP::UserAgent;

my $user = 'test';
my $pass = 'test';

my $content = '5: \'DGN\' (6: \'LE_ART\')
 
        5 \'PILOTES\': UL(\'DGN\'), FS(SUJ), CAT(N, D) .
        6 \'*LES\': UL(\'LE_ART\'), FS(DES), CAT(V, D, S) .
';
my $filename = 'toto.ao';

my $ua  = LWP::UserAgent->new();
my $url = 'http://localhost:8888/arianey/api/users/' . $user . '/ao/' . $filename;
my $header = HTTP::Headers->new();
$header->authorization_basic($user, $pass);
my $req = HTTP::Request->new('POST', $url, $header,$content);
my $res = $ua->request($req);
if ($res->is_success) {
	print $res->content;
}
else {
	print STDERR $res->status_line, "\n";
}
print "\n";

$url = 'http://localhost:8888/arianey/api/compiler/';
$req = HTTP::Request->new('POST', $url, HTTP::Headers->new(),$content);
$res = $ua->request($req);
if ($res->is_success) {
	print $res->content;
}
else {
	print STDERR $res->status_line, "\n";
}
print "\n";
