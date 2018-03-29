Description
=============

Jibiki is a generic platform for managing lexical resources online.
Any resource in XML format can be imported and then queried and edited online.
The platform uses the Enhydra object web framework and Postgres database for the data layer.

For running examples, see:
- http://www.estfra.ee/
- http://papillon.imag.fr/
- http://jibiki.fr/
- http://motamot.imag.fr/

There is also a REST API for remote programming. See
http://papillon.imag.fr/Api.po

Installation
=============

The easiest way to install is to use the dockerfiles.

The jibiki docker image is built from openjdk official image: https://hub.docker.com/_/openjdk/

The postgres4jibiki docker image is built from postgresql latest official image.
See https://github.com/mangeot/postgres4jibiki/ for more information.

The ipolex docker image is built from php latest official image.
See https://github.com/mangeot/ipolex/ for more information.

Getting the latest docker images
-------------
    docker pull mangeot/ipolex
    docker pull mangeot/postgres4jibiki
    docker pull mangeot/jibiki

Or building from the git repos
-------------
    docker build -t mangeot/ipolex github.com/mangeot/ipolex
    docker build -t mangeot/postgres4jibiki github.com/mangeot/postgres4jibiki
    docker build -t mangeot/jibiki github.com/mangeot/jibiki

Running the docker images
-------------
Create a directory on your machine for storing ipolex and postgres data

    mkdir -p /Users/mangeot/docker/ipolex
    mkdir -p /Users/mangeot/docker/postgresData
    
Launch the 3 containers
    
    docker run --name ipolex -p 8888:80 --volume /Users/mangeot/docker/ipolex:/var/www/html/Dicos -d mangeot/ipolex 
    docker run --name jibiki-database --volume /Users/mangeot/docker/postgresData:/var/lib/postgresql/data -d mangeot/postgres4jibiki postgres
    docker run --name jibiki -p 8999:8999 --link jibiki-database:postgres --volume /Users/mangeot/docker/ipolex:/ipolex -d mangeot/jibiki
    
Using the apps
-------------

For iPolex, open http://localhost:8888 in your favorite browser;

For jibiki, open http://localhost:8999 in your favorite browser.

