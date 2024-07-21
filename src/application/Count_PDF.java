package application;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDFont;
import org.apache.pdfbox.pdmodel.font.PDType0Font;

class Count_PDF {
	void run(File inPath, File outPath, ArrayList<Count_item> item_array) {
		String text = inPath.getName() + Controller.newline + Controller.newline; 
		int i = 0;
		for(Count_item item: item_array) {
			text += item.day + Controller.newline;
			text += "LINE: " + item.line + Controller.newline;
			text += "CUT: " + item.cut + Controller.newline + Controller.newline;
			i += item.cut;
		}
		text += "合計: " + i;
		System.out.println(text);
		Controller.scpane_text.setValue(text);
		try {
            createPDF(text,"Result.pdf");
            moveFile.run("Result.pdf", outPath.toString());
        } catch (IOException e) {
            e.printStackTrace();
            Controller.scpane_error.setValue("Result.pdfの作成中にエラーが起きました。" + Controller.newline);
        }
	}
	
	private InputStream getFileAsIOStream(final String fileName) 
    {
        InputStream ioStream = this.getClass()
            .getClassLoader()
            .getResourceAsStream(fileName);
        
        if (ioStream == null) {
            Controller.scpane_error.setValue("フォント'"+Controller.font+"'が見つかりませんでした。" + Controller.newline);
            throw new IllegalArgumentException(fileName + " is not found");
        }
        return ioStream;
    }
	
	public void createPDF(String text, String filePath) throws IOException {
        PDDocument document = new PDDocument();
        PDPage page = new PDPage();
        document.addPage(page);
        
        final InputStream is = getFileAsIOStream(Controller.font);
        PDFont font = PDType0Font.load(document, is );

        PDPageContentStream contentStream = new PDPageContentStream(document, page);
        contentStream.setFont(font, 12);
        contentStream.beginText();
        contentStream.setLeading(14.5f);
        contentStream.newLineAtOffset(50, 750);

        String[] lines = text.split(Controller.newline);
        for (String line : lines) {
            contentStream.showText(line);
            contentStream.newLine();
        }

        contentStream.endText();
        contentStream.close();

        document.save(filePath);
        document.close();
    }
}