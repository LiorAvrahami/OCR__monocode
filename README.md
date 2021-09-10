# OCR_momocode

## introduction
in this project I implement a fully functional cross-correlation based *optical character recognition* program. ** this program is limited to monospace fonts **, and has a hard emphasis on precision. the goal is for this program to, with high certainty, be able to read a printed piece of paper full of gibberish, weird characters, and multiple consecutive spaces, and save it to a .txt file on the computer.   
this program dose require some set up. in order for it to be able to parse some printed text with some font, and some scan resolution, a calibration page must be printed, scanned, and given to the program. the program will create a database file, for this specific font, and scan settings.

## compared to other ocr programs

compared to other ocr programs I found, like tesseracts, this program is different in that it is **not** based on ML, and has the mentioned emphasis on precision 100% of the time, even when dealing with gibberish and non letter characters. from what I gathered, the ML based, ocr programs, like tesseract, are incredibly powerful in the sense that they operate under bad snr ratios, and are incredibly flexible in the since that they operate well with many types of font size, font resolution, and font type (including non-monospaced fonts). this gives them abilities like understanding the text written on a stop sign in the middle of the street. on the other hand, they preform terribly when asked to operate reliably on large text files with gibberish and multiple consecutive spaces. tesseract for example, has a built-in dictionary, and it likes to correct spelling. also, tesseract is incapable keeping track of indentation formatting, it shrinks any set of consecutive spaces and tabs to be a single space.

## transporting text file systems

this project was built with the capability of transporting complete text file systems between computers that are not connected. 
this is done by mapping the howl file system to a single text file, which can be printed, and then scanned by a scanner on the target computer. this program reads and interprets the scanned papers, and remaps the text back to separate files in their original directory hierarchy.   
> currently, the way this is implemented limits the use of '$' in a line (you can't start a line with $$ and then have $$ show up again in that line) but this can easily be fixed by using some unsupported unique character instead of '$'. because of backward compatibility issues, this is not yet fixed.

## usage
```
python \[path directories\]/OCR_monocode [atributes]

atributes:
--exportfilesystem:
   explanation: if this atribute is added then no OCR will be used, intead, the selected input file or folder that represents the root of a file system, will be converted to a single .txt file that can be printed, and can later be scanned and mapped back to the origional file system hyrarchy

-ip, --inputpath:
   -ip [path to folder, or path to single file]
   example:
      python OCR_monocode -ip ../folder_with_images

-op, --outputpath:
   -op [path to to put programs output]
   example:
      python OCR_monocode -op ../out_put_folder

-fs, --fontselect:
   -fs {[font name] [font size] [scanner resolution]dpi}
   example:
      python OCR_monocode -fs {Consols 8 600dpi}

-hf, --hasframe:
   -hf [true or false, case insensative]
   explanation: there is an option to surround the paper with a frame of X, to increase percision. if using "--exportfilesystem" then the correct frame will be automatically added to the text files. this frame will not be in the final .txt files.
```
if some attribute other than exportfilesystem is not given, a nice prompt-toolkit based UI will ask you for this information. it is recommended to use this UI when possible instead of the command line arguments for the sole reason that it is nicer.
> if -fs is not given, the UI will ask you to select a font out of all options available (the options for which a calibration file was made). you can also choose the option "find out for me", which will have the computer run with all possible font settings, and use the one with the best fit.  

*it is usually recommended to print and scan files with a frame when possible.
