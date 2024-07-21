package application;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.pdmodel.font.PDType0Font;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;

class MakePDF {
	void run(ArrayList<pdf_item> pdf_list, File inPath, File outPath) {
		Controller.workin_hard.set("MakePDF");
		Controller.indicator.set(0 +" / "+ pdf_list.size());
		File dstFile = new File(outPath + Controller.separator + "PDF");
		if(!dstFile.exists()){
            dstFile.mkdirs();
		}
		
		int count = 1;
		for(pdf_item pi: pdf_list) {
			Controller.indicator.set(count +" / "+ pdf_list.size());
			count++;
			try {
			      PDDocument document = new PDDocument();
			      PDRectangle rec = new PDRectangle();
					rec.setUpperRightX(0);
					rec.setUpperRightY(0);
					rec.setLowerLeftX((float) 571.2);
					rec.setLowerLeftY((float) 808.32);
			      PDPage page = new PDPage(rec);

				  final InputStream is = getFileAsIOStream(Controller.font);
				    	
			      PDFont font = PDType0Font.load(document, is );
			      
			      PDImageXObject image = null;
			      PDPageContentStream contents = null;
			      String imgname = null;
			      
			      int i2 = 0;
			      for(File f : pi.jpg) {
			    	  Controller.scpane_text.set(f.toString().replace(inPath.toString(), ""));
			    	  
			    	  if(!(f.toString().equals(" .jpg"))) {
			    		  image = PDImageXObject.createFromFile(f.toString(), document);
			    		  imgname = f.getName().substring(0,f.getName().lastIndexOf('.'));
			    	  }
			         switch(i2){
			         case 0:
			        	 contents = new PDPageContentStream(document, page, PDPageContentStream.AppendMode.OVERWRITE, true);
			        	 contents.beginText();
					      contents.setFont(font, 12);
					      contents.newLineAtOffset(480, 780);
					      contents.showText(pi.day);
					      contents.endText();
			        	 document.addPage(page);

			        	 if(f.toString().equals(" .jpg")) {
			        		 
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(50, 640);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 55, 655, 120, 120);
				           i2++;
				       break;
			         case 1:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(221, 640);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 226, 655, 120, 120);
				           i2++;
			           break;
			         case 2:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(393, 640);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 396, 655, 120, 120);
				           i2++;
			           break;
			         case 3:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(50, 487);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 55, 502, 120, 120);
				           i2++;
			           break;
			         case 4:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(221, 487);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 226, 502, 120, 120);
				           i2++;
				       break;
			         case 5:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(393, 487);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 396, 502, 120, 120);
				           i2++;
			           break;
			         case 6:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(50, 334);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 55, 349, 120, 120);
				           i2++;
			           break;
			         case 7:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(221, 334);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 226, 349, 120, 120);
				           i2++;
			           break;
			         case 8:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(393, 334);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 396, 349, 120, 120);
				           i2++;
				       break;
			         case 9:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(50, 181);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 55, 196, 120, 120);
				           i2++;
			           break;
			         case 10:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(221, 181);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 226, 196, 120, 120);
				           i2++;
			           break;
			         case 11:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(393, 181);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 396, 196, 120, 120);
				           i2++;
			           break;
			         case 12:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(50, 28);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 55, 43, 120, 120);
				           i2++;
				       break;
			         case 13:
			        	 if(f.toString().equals(" .jpg")) {
			        		 i2++;
						       break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(221, 28);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 226, 43, 120, 120);
				           i2++;
			           break;
			         case 14:
			        	 if(f.toString().equals(" .jpg")) {
			        		 contents.close();
				        	 page = new PDPage(rec);
				        	 i2 = 0;
				           break;
			        	 }
				           contents.beginText();
						      contents.setFont(font, 12);
						      contents.newLineAtOffset(393, 28);
						      contents.showText(imgname);
						      contents.endText();
			        	 contents.drawImage(image, 396, 43, 120, 120);
			        	 contents.close();
			        	 page = new PDPage(rec);
			        	 i2 = 0;
			           break;
			       }
			      }
			      if(contents != null) {
			      contents.close();
			      }
			      document.save(dstFile + Controller.separator + pi.day + ".pdf");
			      document.close();
			    }
			    catch (IOException e) {
			    	Controller.scpane_error.set("PDFの作成中にエラーが発生しました。");
			    	e.printStackTrace();
			    }
		}
	}
	
	private InputStream getFileAsIOStream(final String fileName) {
        InputStream ioStream = this.getClass()
            .getClassLoader()
            .getResourceAsStream(fileName);
        
        if (ioStream == null) {
            Controller.scpane_error.setValue("フォント'"+Controller.font+"'が見つかりませんでした。" + Controller.newline);
            throw new IllegalArgumentException(fileName + " is not found");
        }
        return ioStream;
    }
}