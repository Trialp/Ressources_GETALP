#!/usr/bin/env perl

###################################################
#
# This script is used for running several instances of Jibiki on the same server.
# It is undocumented. Use at your own risks
# It is used with the sites.conf.in file.
#
##################################################

use warnings;
use strict;
#`../toolsforjibiki/enhydra5.1/bin/ant quick`;
`../../../toolsforjibiki/enhydra5.1/bin/ant quick`;

local $/ = undef;
open FILE, "output/conf/Papillon.conf" or die "Couldn't open file: $!";
binmode FILE;
my $PapillonConf = <FILE>;
close FILE;

local $/ = "\n";
my $file = 'sites.conf';
open my $info, $file or die "Could not open $file: $!";

my %sites = ();
while( my $line = <$info>)  {
	chomp($line);
	if ($line ne '' && $line !~ /^#/) {
		my @ligne = split(/\t+/,$line);
		$sites{shift(@ligne)} = \@ligne;
	}
}
close $info;


my $application = 'Application';
my $connection = 'Connection';
my $channel = 'Channel';
my $filename = "output/conf/multimono.conf";
open(my $multimonoConf, '>>', $filename) or die "Could not open file '$filename' $!";

my $nbsite = 1;
foreach my $site (keys %sites)
{
	my $sitelc = lc($site);
	my $siteConf = $PapillonConf;
	my @site = @{ $sites{$site} };
	$siteConf =~ s/Application\.Prefix = "\/[^"]*"/Application.Prefix = "\/$sitelc"/;
	$siteConf =~ s/"PapillonLoginCookie"/"${site}LoginCookie"/;
	$siteConf =~ s/"fr.imag.clips.papillon.presentation.xhtml"/"fr.imag.clips.papillon.presentation.xhtml$sitelc"/;
	$siteConf =~ s/"jdbc:postgresql:[^"]*"/"jdbc:postgresql:\/\/localhost\/$sitelc"/;
	
	my $filename = "output/conf/$site.confm";
	open(my $fh, '>', $filename) or die "Could not open file '$filename' $!";
	print $fh $siteConf;
	close $fh;
	my $applicationSite = $application . '.' . $site . '.';
	print $multimonoConf "\n### $site\n";
	print $multimonoConf $applicationSite, "ConfFile = $site.confm\n";
	print $multimonoConf $applicationSite, "Description = \"$site\"\n";
	print $multimonoConf $applicationSite, "Running = yes\n";
	print $multimonoConf $applicationSite, "defaultSessionTimeOut = 30\n\n";
	
	my $connectionSite = $connection . '.http' .  $nbsite . '.';
	print $multimonoConf $connectionSite, "Type = http\n";
	print $multimonoConf $connectionSite, "Port = $site[0]\n\n";
	
	my $channelSite = $channel . '.http' .  $nbsite . '.channel.';
	print $multimonoConf $channelSite, "Servlet = $site\n";
	print $multimonoConf $channelSite, "Url = /$sitelc\n";
	print $multimonoConf $channelSite, "Enabled = yes\n\n";

	$nbsite++;
	print "Site : $site : ", join(' ',@site)," done\n";
}
close $multimonoConf;
