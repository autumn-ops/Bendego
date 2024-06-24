package application;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

class Count implements Runnable {
	
	private final String mode;
    private final File inPath;
    private final File outPath;

    public Count(String mode, File inPath, File outPath) {
        this.mode = mode;
        this.inPath = inPath;
        this.outPath = outPath;
    }
    
    ArrayList<Count_item> item_array = new ArrayList<Count_item>();
 	ArrayList<File> file_array = new ArrayList<File>();
 	List<File> l_list;
 	int cut;
	
	@Override
	public void run() {
		Controller.workin_hard.set("Count");
		//号数リスト
		for (File tmpFile : inPath.listFiles()) {
    		if(tmpFile.isHidden() == false ) {
    			if(tmpFile.isDirectory()){
    				file_array.add(tmpFile);
    			}
    		}
        }
		file_array.sort(null);
		
		//
		for(File f: file_array) {
			l_list = new ArrayList<File>();
			cut = 0;
			
			try {
	            Files.walkFileTree(f.toPath(), new SimpleFileVisitor<Path>() {
	            	@Override
	                public FileVisitResult visitFile(Path file,
	                        BasicFileAttributes attrs) throws IOException {
	            		if(!file.toFile().isHidden() || !file.getFileName().toString().equals("CaptureOne") || file.toFile().isFile()) {
	            			if(!l_list.contains(file.toFile().getParentFile())) {
	            				//行数取得
	            				l_list.add(file.toFile().getParentFile());
	            			 	DumpFile df = new DumpFile();
	            			 	cut += df.run(file.toFile().getParentFile()).size();
	            			}
	            		}
	            		return FileVisitResult.CONTINUE;
	            	}
	         	  });
	      	    } catch (IOException e) {
	      		    e.printStackTrace();
	      		  Controller.scpane_error.setValue("ファイルを読み込む際にエラーが起きました。\n"
	      		  		+ e.toString());
	      	    }

			//号・行・コマ数の出力
			Count_item item = new Count_item(f.getName(), l_list.size(), cut);
			item_array.add(item);
		}
		Controller.workin_hard.set("結果の作成");
		Count_PDF cp = new Count_PDF();
		cp.run(inPath, outPath, item_array);
		Controller.workin_hard.set("終了");
	}
}