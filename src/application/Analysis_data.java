package application;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PushbackInputStream;
import java.util.Arrays;

import org.apache.poi.EncryptedDocumentException;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.poifs.filesystem.FileMagic;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;


class Analysis_data {
	private static Workbook excel;

    private static final int MAX_PATTERN_LENGTH = 44;
    
	int getLine(File f) {
		
		String sheetName = "原稿①";
		if(!Controller.txtf.equals("")) {
			sheetName = Controller.txtf;
		}

	    int startline = 3;
	    
	    int num = 0;

		try {
			FileInputStream inst = new FileInputStream(f);
			
			PushbackInputStream inp = new PushbackInputStream(inst, MAX_PATTERN_LENGTH);
	        byte[] data = new byte[MAX_PATTERN_LENGTH];
	        inp.read(data, 0, MAX_PATTERN_LENGTH);
	        inp.unread(data);
	        FileMagic fm = FileMagic.valueOf(data);
	        if (FileMagic.OOXML==fm){
	        	excel = new XSSFWorkbook(inp);
	        }else if(FileMagic.OLE2==fm){
	        	excel = new HSSFWorkbook(inp);
	        }else {
	        	excel = WorkbookFactory.create(inp);
	        }
	        
        // シート名がわかっている場合
	    Sheet sheet = excel.getSheet(sheetName);
        int[] page = sheet.getRowBreaks();

        int item_index = 0;
        int goods_index = 0;
        int code_index = 0;

        int skipnum = 0;
        
        int score;
        
        for(int i=startline; i<=page[page.length-1]; i++) {

    		Row row = sheet.getRow(i);//複数行使用している場合は、一番上の行
    		
    		for (Cell cell : row) {
                int rowIndex = cell.getRowIndex(); // 行番号
                int colIndex = cell.getColumnIndex(); // 列番号
                String cellValue = getValue(cell);
                
                //品名
                if ("品名".equals(cellValue)) {
                	item_index = colIndex;
                }
                //商品記号
                else if ("商品記号".equals(cellValue)) {
                	goods_index = colIndex;
                }
                //アイテムコード
                else if ("アイテムコード（13桁）".equals(cellValue)) {
                	code_index = colIndex;
                	skipnum = rowIndex+1;
                }
                
            }
    		
    		if(skipnum >= i) {
    			continue;
    		}
        	
    		score = 0;
    		
			String item = getValue(row.getCell(item_index)); //商品名
			String goods = getValue(row.getCell(goods_index));//商品記号
			String code = getValue(row.getCell(code_index));   //アイテムコード

	        if(item != "") {
	        	score += 2;
	        }
	        if(goods != "") {
	        	score += 1;
	        }
	        if(code != "") {
	        	score += 1;
        	}
	        if(score > 1) {
	        	num++;
	        }
        	
        	//ページの終端
        	else if(contains(page, i)) {
        		i += 5;
        	}
        }
        excel.close();
		} catch (EncryptedDocumentException | IOException e) {
			Controller.scpane_text.set("Excelデータの読み込みに失敗しました。");
			e.printStackTrace();
		}
		return num;
	}
	
	public static boolean contains(final int[] page, final int key) {
        return Arrays.stream(page).anyMatch(i -> i == key);
    }

    static String getValue(Cell cell) {
        if (cell == null) {
            return "";
        }
        CellType cType = cell.getCellType();
        switch (cType) {
            case STRING:
                return cell.getStringCellValue().replace("\n", "").replace(" ", "");
            case NUMERIC:
                return String.valueOf(cell.getNumericCellValue());
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                return cell.getCellFormula();
            default:
                return "";
        }
    }
}