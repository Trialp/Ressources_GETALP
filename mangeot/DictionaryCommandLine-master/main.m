// main.m
//
// Copyright (c) 2015 Mathieu Mangeot
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#import <Foundation/Foundation.h>

#import "TTTDictionary.h"

void NSPrint (NSString *str)
{
    NSError *error;
    [str writeToFile: @"/dev/stdout" atomically: NO encoding: NSUTF8StringEncoding error:&error];
}

void NSPrintError (NSString *str)
{
    NSError *error;
    [str writeToFile: @"/dev/stderr" atomically: NO encoding: NSUTF8StringEncoding error:&error];
}

void PrintHelpMessage (NSSet *dicts) {
    NSPrintError([NSString stringWithFormat:@"Usage: %s [-d dictionary] [-l lang] [-f format] \"word\".\n\n",[[[NSProcessInfo processInfo] processName] UTF8String]]);
    NSPrintError([NSString stringWithFormat:@"%li dictionaries are available:\n", [dicts count]]);
    for (TTTDictionary *dictionary in dicts) {
        NSPrintError([NSString stringWithFormat:@"%@ => %@\n", dictionary.shortName, dictionary.name]);
    }
    NSPrintError(@"all dictionaries are queried by default\n");
    NSPrintError(@"\n9 languages are available:\n");
    NSPrintError(@"de => German\n");
    NSPrintError(@"en => English\n");
    NSPrintError(@"es => Spanish\n");
    NSPrintError(@"fr => French\n");
    NSPrintError(@"it => Italian\n");
    NSPrintError(@"ja => Japanese\n");
    NSPrintError(@"kr => Korean\n");
    NSPrintError(@"nl => Dutch\n");
    NSPrintError(@"zh => Simplified Chinese\n");
    NSPrintError(@"default language is not specified\n");
 
    NSPrintError(@"\n6 format are available:\n");
    NSPrintError(@"headword => prints only the headword\n");
    NSPrintError(@"definition => prints the headword and the definition in text format\n");
    NSPrintError(@"definitionHTML => prints the headword and the definition in HTML format\n");
    NSPrintError(@"text => prints the headword and the definition in text format\n");
    NSPrintError(@"HTML => prints the whole entry content in HTML format\n");
    NSPrintError(@"HTMLCSS => prints the whole entry content in HTML+CSS format\n");
    NSPrintError(@"default format is text\n");

    NSPrintError(@"\n");
 }

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        NSSet *allDictionaries = [TTTDictionary availableDictionaries];
        NSDictionary *langDictNames = [TTTDictionary langDictNames];
        NSError *error = nil;
        NSString *format = @"text";
        NSString *lang = @"";
        NSString *dict = @"";

        
        //check whether argument is there
        int nbargs = (int) [[[NSProcessInfo processInfo] arguments] count];
        
        if (nbargs % 2 == 1)
        {
           PrintHelpMessage(allDictionaries);
           exit(1);
        }
        int i = 1;
        while (i < nbargs-1) {
            NSString *arg = [[[NSProcessInfo processInfo] arguments] objectAtIndex: i];
            if ([arg  isEqual: @"-l"]) {
                i++;
                lang = [[[NSProcessInfo processInfo] arguments] objectAtIndex: i];
            }
            else if ([arg isEqual: @"-d"]) {
                i++;
                dict = [[[NSProcessInfo processInfo] arguments] objectAtIndex: i];
            }
            else if ([arg isEqual: @"-f"]) {
                i++;
                format = [[[NSProcessInfo processInfo] arguments] objectAtIndex: i];
            }
            i++;
        }
        
        NSString *term = [[[NSProcessInfo processInfo] arguments] objectAtIndex: i];
        if (!term)
        {
            NSPrintError([NSString stringWithFormat:@"Agument is missing: %s, error: %s", [term UTF8String], [[error localizedDescription] ? [error localizedDescription] : [error description] UTF8String]]);
            exit(1);
        }
        
        //NSLog(@"Format: %@ lang: %@, dict: %@, term: %@\n", format, lang, dict, term);
        
        
        NSSet *dictSet = allDictionaries;
        if (![lang isEqual: @""]) {
            NSSet *langSet = langDictNames[lang];
            dictSet = [allDictionaries objectsPassingTest:^BOOL(id obj, BOOL *stop) {
                if ([langSet containsObject:((TTTDictionary *)obj).name]) {
                    return YES;
                } else {
                    return NO;
                }
            }];
       }
        else if (![dict isEqual: @""]) {
            dictSet = [allDictionaries objectsPassingTest:^BOOL(id obj, BOOL *stop) {
                if ([((TTTDictionary *)obj).shortName caseInsensitiveCompare: dict] == NSOrderedSame) {
                    return YES;
                } else {
                    return NO;
                }
            }];
        }
        
        for (TTTDictionary *dictionary in dictSet) {
            //NSLog(@"%@\n", dictionary.name);
            for (TTTDictionaryEntry *entry in [dictionary entriesForSearchTerm:term]) {
                if ([format caseInsensitiveCompare: @"headword"] == NSOrderedSame) {
                    NSPrint([NSString stringWithFormat:@"%@:%@\n", dictionary.name, entry.headword]);
                }
                else if ([format caseInsensitiveCompare: @"definition"] == NSOrderedSame) {
                    NSPrint([NSString stringWithFormat:@"%@:\n%@\n", dictionary.name, entry.definition]);
                }
                else if ([format caseInsensitiveCompare: @"text"] == NSOrderedSame) {
                    NSPrint([NSString stringWithFormat:@"%@:\n%@\n", dictionary.name, entry.text]);
                }
                else if ([format caseInsensitiveCompare: @"definitionHTML"] == NSOrderedSame) {
                    NSPrint(entry.definitionHTML);
                }
                else if ([format caseInsensitiveCompare: @"HTMLCSS"] == NSOrderedSame) {
                    NSPrint(entry.HTMLCSS);
                }
                else if ([format caseInsensitiveCompare: @"HTML"] == NSOrderedSame) {
                    NSPrint(entry.HTML);
                }
            }
       }
    }
    return 0;
}
