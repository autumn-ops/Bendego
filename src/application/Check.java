package application;

import java.awt.MediaTracker;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.imageio.ImageIO;

class Check {
	ArrayList<File> run(ArrayList<File> list) {
		Controller.workin_hard.set("Check");
    	
		String regex = "(\\s)";
        Pattern p = Pattern.compile(regex);
        
        String regex1 = "コピー";
        Pattern p1 = Pattern.compile(regex1);
        String regex2 = " ";
        Pattern p2 = Pattern.compile(regex2);
        
        int i = 1;
    	for(File f: list) {
    		
    		Controller.scpane_text.set(f.getParentFile().getParentFile().getParentFile().getName() 
    				+"_"+ f.getParentFile().getParentFile() .getName()
    				+"_"+ f.getParentFile().getName() +"_"+ f.getName());
    		
    		Controller.indicator.set(i + " / " + list.size());
    		
    		try {
				MediaTracker tracker = new MediaTracker(null);
				
				BufferedImage img = ImageIO.read(f);

				tracker.addImage(img, 0);
		    	
			} catch (IOException e) {
				Controller.scpane_error.set("ERROR: " + f + "のカラースペースは、CMYK の可能性があります。" + Controller.newline);
			}
    		
            Matcher m = p.matcher(f.getName());
            if (m.find()){
            	
            	String sre  = " ";
            	Pattern pre = Pattern.compile(sre);

            	Matcher mre = pre.matcher((
            			f.getName().substring(0,f.getName().lastIndexOf(" "))
            			+f.getName().toString().substring
    					(f.getName().toString().lastIndexOf("."))).replaceAll(" ", ""));
            	
            	File ref = new File(f.getParent() + Controller.separator + mre.replaceAll(""));
            	
            	if(ref.exists()) {
            		
            		int num = 1;
            		File reref;
            		
            		while (true){
            			reref = new File( (ref.getParent() + Controller.separator
            					+ ref.getName().toString().substring
            					(0,ref.getName().toString().lastIndexOf("_")+1)
            					+ num
            					+ f.getName().toString().substring
            					(f.getName().toString().lastIndexOf("."))).replace("＿","_") );
            			
            			if(!reref.exists()) {
            			break;
            			}
            			
            		  num++;
            		}
            		f.renameTo(reref);
            	}else {
                    f.renameTo(ref);
            	}
            }
            
            Matcher m1 = p1.matcher(f.getName());
            Matcher m2 = p2.matcher(f.getName());
            if (m1.find() || m2.find()){

            	File reft = new File(f.toString().replace(f.getName(),
            			f.getName().replace("＿","_").replace("のコピー", "").replace("コピー", "").replace("の", "").replace(" ", ""))
            			);
            	
            	File ref = new File(reft.getParentFile() + Controller.separator + reft.getName().substring(0, reft.getName().indexOf(".")) + ".jpg");
            	
            	if(ref.exists()) {
            		
            		int num = 1;
            		File reref;
            		
            		while (true){
            			reref = new File( (ref.getParent() + Controller.separator
            					+ ref.getName().toString().substring
            					(0,ref.getName().toString().lastIndexOf("_")+1)
            					+ num
            					+ f.getName().toString().substring
            					(f.getName().toString().lastIndexOf("."))).replace("＿","_") );
            			
            			if(!reref.exists()) {
            			break;
            			}
            			
            		  num++;
            		}
            		f.renameTo(reref);
            	}else {
                    f.renameTo(ref);
            	}
            }
            i++;
    	}
		return list;
	}
}