

create table contributionlog 
(
/* class ContributionLog*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    author VARCHAR (255) NOT NULL ,
    groups VARCHAR (255) NOT NULL ,
    volume VARCHAR (255) NOT NULL ,
    sourceLanguage VARCHAR (255) NOT NULL ,
    headword VARCHAR (255) NOT NULL ,
    contributionId VARCHAR (255) NOT NULL ,
    entryId VARCHAR (255) NOT NULL ,
    date TIMESTAMP NOT NULL ,
    status VARCHAR (255) NOT NULL
);


CREATE INDEX contributionlog_author_idx ON contributionlog (author);
CREATE INDEX contributionlog_volume_idx ON contributionlog (volume);
CREATE INDEX contributionlog_headword_idx ON contributionlog (headword);
CREATE INDEX contributionlog_contributionid_idx ON contributionlog (contributionId);
CREATE INDEX contributionlog_entryid_idx ON contributionlog (entryId);
CREATE INDEX contributionlog_status_idx ON contributionlog (status);




create table dictionaries 
(
/* class Dictionary*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    fullName VARCHAR (255) NOT NULL ,
    name VARCHAR (255) NOT NULL ,
    category VARCHAR (255) NOT NULL ,
    type VARCHAR (255) NOT NULL ,
    domain VARCHAR (255) NOT NULL ,
    legal VARCHAR (255) NOT NULL ,
    access VARCHAR (255) NOT NULL ,
    sourceLanguages VARCHAR (255) NOT NULL ,
    targetLanguages VARCHAR (255) NOT NULL ,
    xmlCode TEXT NOT NULL
);


CREATE UNIQUE INDEX dictionaries_fullname_idx ON dictionaries (fullname);
CREATE UNIQUE INDEX dictionaries_name_idx ON dictionaries (name);




create table foksentry 
(
/* class FoksEntry*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    headword VARCHAR (255) ,
    reading VARCHAR (255) ,
    id INTEGER ,
    score REAL
);


CREATE INDEX foksentry_reading_idx ON foksentry (reading);




create table groups 
(
/* class Group*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    name VARCHAR (255) NOT NULL ,
    password BYTEA NOT NULL ,
    users TEXT ,
    admins TEXT
);


CREATE UNIQUE INDEX groups_name_idx ON groups (name);




create table informationdocument 
(
/* class InformationDocument*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    author TEXT ,
    owner VARCHAR (255) NOT NULL ,
    reference TEXT ,
    creationDate DATE NOT NULL ,
    section VARCHAR (255) ,
    title TEXT
);






create table informationfiles 
(
/* class InformationFile*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    filename TEXT NOT NULL ,
    filetype TEXT NOT NULL ,
    document DECIMAL (19,0) NOT NULL ,
    filecode TEXT ,
    isIndexFile CHAR (1) NOT NULL ,
    language VARCHAR (5) NOT NULL ,    
    CONSTRAINT informationfiles_document FOREIGN KEY (document) REFERENCES informationdocument (ObjectId)
);






create table jibikiversion 
(
/* class JibikiVersion*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    number INTEGER NOT NULL
);






create table messages 
(
/* class Message*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    date DATE NOT NULL ,
    msg TEXT NOT NULL ,
    msgid VARCHAR (127) NOT NULL ,
    subject VARCHAR (255) NOT NULL ,
    author VARCHAR (255) NOT NULL ,
    authorAddress VARCHAR (255) NOT NULL ,
    thread VARCHAR (255) NOT NULL
);


CREATE UNIQUE INDEX messages_msgid_idx ON messages (msgid);




create table querylogs 
(
/* class QueryLog*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    date DATE NOT NULL ,
    login VARCHAR (255) NOT NULL ,
    formName VARCHAR (50) NOT NULL ,
    prefLang VARCHAR (3) NOT NULL ,
    query VARCHAR (255) NOT NULL ,
    results TEXT ,
    srcLang VARCHAR (3) NOT NULL ,
    trgLangs VARCHAR (512) NOT NULL ,
    dicts VARCHAR (1024) NOT NULL ,
    strategy VARCHAR (255) NOT NULL ,
    queryString TEXT NOT NULL
);


CREATE INDEX querylogs_login_idx ON querylogs (login);
CREATE INDEX querylogs_formName_idx ON querylogs (formName);
CREATE INDEX querylogs_query_idx ON querylogs (query);
CREATE INDEX querylogs_queryString_idx ON querylogs (queryString);




create table users 
(
/* class User*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    name VARCHAR (255) NOT NULL ,
    login VARCHAR (255) NOT NULL ,
    password VARCHAR (255) NOT NULL ,
    email VARCHAR (255) NOT NULL ,
    lang VARCHAR (3) ,
    groups VARCHAR (255) ,
    credits INTEGER NOT NULL ,
    xmlCode TEXT ,
    creationDate DATE ,
    modificationDate DATE
);


CREATE UNIQUE INDEX users_name_idx ON users (name);
CREATE UNIQUE INDEX users_login_idx ON users (login);




create table volumes 
(
/* class Volume*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    name VARCHAR (255) NOT NULL ,
    dictname VARCHAR (255) NOT NULL ,
    dbname VARCHAR (255) NOT NULL ,
    location VARCHAR (255) NOT NULL ,
    sourceLanguage VARCHAR (3) NOT NULL ,
    targetLanguages VARCHAR (255) NOT NULL ,
    volumeRef VARCHAR (255) NOT NULL ,
    xmlCode TEXT NOT NULL ,
    xmlSchema TEXT ,
    templateItf TEXT ,
    templateEntry TEXT ,
    reverse CHAR (1) NOT NULL ,
    entries INTEGER NOT NULL
);


CREATE UNIQUE INDEX volumes_name_idx ON volumes (name);
CREATE INDEX volumes_dictname_idx ON volumes (dictname);
CREATE INDEX volumes_dbname_idx ON volumes (dbname);




create table xslsheets 
(
/* class XslSheet*/
    ObjectId DECIMAL (19,0) NOT NULL PRIMARY KEY ,
    ObjectVersion INTEGER  NOT NULL ,
    name VARCHAR (255) NOT NULL ,
    dictionaryName VARCHAR (255) NOT NULL ,
    volumeName VARCHAR (255) NOT NULL ,
    description VARCHAR (255) NOT NULL ,
    code TEXT NOT NULL ,
    defaultxsl CHAR (1) NOT NULL ,
    externalxsl CHAR (1) NOT NULL
);







create table objectid
(
    next DECIMAL (19,0) NOT NULL PRIMARY KEY
);
