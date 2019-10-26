package cs764;

public class ConversionException extends Exception{
  public ConversionException() { super(); }
  public ConversionException(String message) { super(message); }
  public ConversionException(String message, Throwable cause) { super(message, cause); }
  public ConversionException(Throwable cause) { super(cause); }

}