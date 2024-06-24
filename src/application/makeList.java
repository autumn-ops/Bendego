package application;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

class makeList{
	static ArrayList<File> list = new ArrayList<>();
	
	String makeName(File file, File inPath, File outPath) throws FileNotFoundException {
		//保存するPDFのパスを作成する
		String dir = outPath + Controller.separator + "原稿PDF";
		dir = file.getParent().replace(inPath.toString(), dir);
		try{
			  Files.createDirectories(Paths.get(dir));
			}catch(IOException e){
			  System.out.println(e);
			}
		String s = file.toString().replace(file.getParent(), dir);
		String name = s.substring(0, s.lastIndexOf(".")) + ".pdf";
		
		return name;
	}
}