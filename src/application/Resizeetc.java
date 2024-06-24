package application;

import java.awt.Image;
import java.awt.color.ICC_ColorSpace;
import java.awt.color.ICC_Profile;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.stream.FileImageOutputStream;

import org.w3c.dom.Element;

import javafx.application.Application;

public abstract class Resizeetc extends Application {

	static void BufferedImage(File f, InputStream is, String savePath) throws IOException{

		int width = 740;
		int height = 740;

		ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
		ImageWriteParam param    = writer.getDefaultWriteParam();

		param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
		param.setCompressionQuality(1f);
		
		
		BufferedImage bimage0 = ImageIO.read(f);
		
		//Resize処理
		BufferedImage resizeImage = new BufferedImage(width, height, bimage0.getType());
		resizeImage.getGraphics().drawImage(
				bimage0.getScaledInstance(width, height, Image.SCALE_AREA_AVERAGING)
				,0, 0, width, height, null);
		//

		//プロファイル処理
		ICC_Profile ip = ICC_Profile.getInstance(is);
		ICC_ColorSpace ics = new ICC_ColorSpace( ip );
		ColorConvertOp cco = new ColorConvertOp(ics, null);
		BufferedImage bimage = cco.filter(resizeImage, null);
		//
		
		RenderedImage image = bimage;

		//DPI処理
		IIOMetadata metadata = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), param);
		Element     tree     = (Element)metadata.getAsTree("javax_imageio_jpeg_image_1.0");
		Element     jfif     = (Element)tree.getElementsByTagName("app0JFIF").item(0);
		jfif.setAttribute("Xdensity", Integer.toString(72));
		jfif.setAttribute("Ydensity", Integer.toString(72));
		jfif.setAttribute("width", Integer.toString(740));
		jfif.setAttribute("height", Integer.toString(740));
		jfif.setAttribute("resUnits", "1"); // In pixels-per-inch units
		metadata.mergeTree("javax_imageio_jpeg_image_1.0", tree);
		//

		FileImageOutputStream output = new FileImageOutputStream(new File(savePath + Controller.separator + f.getName()));
		writer.setOutput(output);
		IIOImage iioImage = new IIOImage((RenderedImage) image, null, metadata);

		writer.write(metadata, iioImage, param);
		writer.dispose();
	}
}