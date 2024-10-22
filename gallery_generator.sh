 #!/bin/bash                                                                                                                                         
                                                                                                                                                     
 # Find all files in the assets directory                                                                                                            
 find assets -type f -print0 | while IFS= read -r -d $'\0' file; do                                                                                  
   # Extract filename and extension                                                                                                                  
   filename=$(basename "$file")                                                                                                                      
   extension="${filename##*.}"                                                                                                                       
                                                                                                                                                     
   # Check if the file is an image                                                                                                                   
   case "$extension" in                                                                                                                              
     png|jpg|jpeg|gif|bmp|svg|webp)                                                                                                                  
       # Generate markdown image tag                                                                                                                 
       echo "![ $filename]($file)" >> assets/README.md
       ;;                                                                                                                                            
     *)                                                                                                                                              
       # Ignore non-image files                                                                                                                      
       echo "Skipping non-image file: $file"                                                                                                         
       ;;                                                                                                                                            
   esac                                                                                                                                              
 done                                                                                                                                                
                                                                                                                                                     
 echo "Gallery generated in assets/README.md"    
