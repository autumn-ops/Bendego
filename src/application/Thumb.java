package application;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

class Thumb implements Runnable {

	private final String mode;
	private final File inPath;
	private final File outPath;

	public Thumb(String mode, File inPath, File outPath) {
		this.mode = mode;
		this.inPath = inPath;
		this.outPath = outPath;
	}

	ArrayList<File> file_list = new ArrayList<File>();
	ArrayList<Day_item> thum_list = new ArrayList<Day_item>();
	ArrayList<String> day_list = new ArrayList<String>();
	
	ArrayList<pdf_item> pdf_list = new ArrayList<pdf_item>();

	@Override
	public void run() {
		Controller.workin_hard.set("Thumb");
		ArrayList<File> list = new ArrayList<File>();

		DumpFile df = new DumpFile();
		list = df.run(inPath);
		if(list.size() == 0) {
			Controller.scpane_error.set("フォルダ内に画像がありません。");
			return;
		}
		
		//PDFファイル用のリスト
		File dir;
		String day;
		String item;
		int cont;
		for(File f: list) {
			if(f.getName().substring(f.getName().lastIndexOf(".")).equals(".jpg")) {
				
				//アイテムコード
				item = f.getName().replace(".", "_");
				item = item.substring(item.indexOf("_")+1, f.getName().length());
				item = item.substring(0, item.indexOf("_", 3));
				
				//号数
				if(f.getParentFile().getName().equals(item)) {
					dir = f.getParentFile().getParentFile();
				}else {
					dir = f.getParentFile();
				}
				
				day = dir.toString().replace(inPath.toString()+Controller.separator, "").replace(Controller.separator, "_");
				
				if(thum_list.size() == 0) {
					thum_list.add(new Day_item(day, item, new ArrayList<File>()));
					thum_list.get(0).jpg.add(f);
					continue;
				}
				if(!day_list.contains(day)) {
					day_list.add(day);
				}
				
				cont = containsOf(thum_list, item);
				if(cont == -1) {
					Collections.sort(thum_list.get(thum_list.size()-1).jpg, new CustomModifiedComparator());
					if(thum_list.get(thum_list.size()-1).jpg.size()%3 != 0) {
						for( int i=thum_list.get(thum_list.size()-1).jpg.size()%3; i<3; i++ ) {
							thum_list.get(thum_list.size()-1).jpg.add(new File(" .jpg"));
						}
					}
					thum_list.add(new Day_item(day, item, new ArrayList<File>()));
					thum_list.get(thum_list.size()-1).jpg.add(f);
					
				}else {
					thum_list.get(cont).jpg.add(f);
				}
			}
		}
		Collections.sort(thum_list.get(thum_list.size()-1).jpg, new CustomModifiedComparator());
		if(thum_list.get(thum_list.size()-1).jpg.size()%3 != 0) {
			for( int i=thum_list.get(thum_list.size()-1).jpg.size()%3; i<3; i++ ) {
				thum_list.get(thum_list.size()-1).jpg.add(new File(" .jpg"));
			}
		}
		
		ArrayList<File> l;
		for(String d: day_list) {
			l = new ArrayList<File>();
			for(Day_item di: thum_list) {
				if(di.day.equals(d)) {
					for(File f: di.jpg) {
						l.add(f);
					}
				}
			}
			pdf_list.add(new pdf_item(d, l));
		}
		
		
		
		
		MakePDF mp = new MakePDF();
		mp.run(pdf_list, inPath, outPath);
		
		
		Controller.workin_hard.set("END");
	}
	
	int containsOf(ArrayList<Day_item> thum_list, String code) {
		for(int i=0; i<thum_list.size(); i++) {
			if(thum_list.get(i).code.equals(code)){
				return i;
			}
		}
		return -1;
	}
	
	class CustomModifiedComparator implements Comparator<Object> {

    	public int compare(Object o1, Object o2) {
    		Long i1 = (long) 0;
    		Long i2 = (long) 0;
    		if(!(o1 == null)||!(o2 == null)) {
    		File f1 = (File)o1;
    		File f2 = (File)o2;
    		
    		String s1 = new String(f1.getName().substring(
    				f1.getName().lastIndexOf("_")+1,
    				f1.getName().lastIndexOf(".")));
    		String s2 = new String(f2.getName().substring(
    				f2.getName().lastIndexOf("_")+1,
    				f2.getName().lastIndexOf(".")));
    		
        	if(s1.equals("c")) {
        		s1 = new String(f1.getName().substring(
        				f1.getName().lastIndexOf("_",
        						f1.getName().lastIndexOf("_")-1)+1,
        				f1.getName().lastIndexOf(".")-2));
        		i1 =- (long) (1);
        	}
        	if(s2.equals("c")) {
        		
        		s2 = new String(f2.getName().substring(
        				f2.getName().lastIndexOf("_",
        						f2.getName().lastIndexOf("_")-1)+1,
        				f2.getName().lastIndexOf(".")-2));
        		i2 =- (long) (1);
        	}
        	if(s1.replaceAll("[^0-9]", "").equals("")) {s1 = "1";}
        	if(s2.replaceAll("[^0-9]", "").equals("")) {s2 = "1";}
        	Long ss1 = Long.parseLong(s1.replaceAll("[^0-9]", "")) *10;
        	Long ss2 = Long.parseLong(s2.replaceAll("[^0-9]", "")) *10;
        	
        	i1 = ss1 + i1;
        	i2 = ss2 + i2;
        	
    		}
    		return (new Long(i1).compareTo(new Long(i2)));
    	}

    	public boolean equals(Object o1, Object o2) {
    		String s1 = "";
    		String s2 = "";
    		if(!(o1 == null)||!(o2 == null)) {
    		File f1 = (File)o1;
    		File f2 = (File)o2;
    		s1 = new String(f1.getName().substring(
    				f1.getName().lastIndexOf("_")+1,
    				f1.getName().lastIndexOf(".")));
    		s2 = new String(f2.getName().substring(
    				f2.getName().lastIndexOf("_")+1,
    				f2.getName().lastIndexOf(".")));

        	if(s1.equals("c")) {
        		s1 = new String(f1.getName().substring(
        				f1.getName().lastIndexOf("_",
        						f1.getName().lastIndexOf("_")-1)+1,
        				f1.getName().lastIndexOf("_")));
        	}
        	if(s2.equals("c")) {
        		s2 = new String(f2.getName().substring(
        				f2.getName().lastIndexOf("_",
        						f2.getName().lastIndexOf("_")-1)+1,
        				f2.getName().lastIndexOf("_")));
        	}
    		}
    		return s1 == s2;
    	}
    }
}

class Day_item {
	String day;  //号数
	String code;  //アイテムコード
	ArrayList<File> jpg;  //画像
 	 
	Day_item(String day, String code,ArrayList<File> jpg) {
		this.day = day;
		this.code = code;
		this.jpg = jpg;
	}
}


class pdf_item {
	String day;  //号数
	ArrayList<File> jpg;  //画像
 	 
	pdf_item(String day, ArrayList<File> jpg) {
		this.day = day;
		this.jpg = jpg;
	}
}