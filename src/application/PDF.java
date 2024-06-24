package application;

import java.io.File;
import java.util.ArrayList;

class PDF implements Runnable {
	
	private final String mode;
    private final File inPath;
    private final File outPath;

    public PDF(String mode, File inPath, File outPath) {
        this.mode = mode;
        this.inPath = inPath;
        this.outPath = outPath;
    }

    ArrayList<File> filelist = new ArrayList<File>();
	  
	@Override
	public void run() {
		Controller.workin_hard.set("PDFに変換");
		ArrayList<File> list = new ArrayList<File>();
		
		//ファイルパスをリストに代入
		if(inPath.isFile()) {
			if(inPath.toString().substring(inPath.toString().lastIndexOf(".")).equals(".xlsx")) {
				list.add(inPath);
			}else {
				Controller.scpane_error.set("読み込めませんでした。 " + inPath.getName());
			}
		}else {
			DumpFile df = new DumpFile();
			list = df.run(inPath);
		}
		
		//.xlsxファイルだけ取り出す
		for(File f: list) {
			if(f.toString().substring(f.toString().lastIndexOf(".")).equals(".xlsx")) {
				filelist.add(f);
			}else {
				Controller.scpane_error.set("読み込めませんでした。 " + f.getName());
			}
		}
		
		filelist.sort(null);
		
		int num = 0;
		int i = 1;
		for(File f: filelist) {
			Controller.indicator.set(i + " / " + filelist.size());
			Controller.scpane_text.set(f.getName());
		    createPDF cp = new createPDF();
			num += cp.run(f, inPath, outPath);
			i++;
		}
		Controller.scpane_text.set("  合計" + num + "行" + Controller.newline);
		Controller.workin_hard.set("終了");
	  }
}