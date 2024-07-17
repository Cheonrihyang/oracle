package polymorphism;

public class LGTV implements TV{
	
	@Override
	public void powerOn() {
		System.out.println("LGTV.... 전원을 켠다");
	}
	@Override
	public void powerOff() {
		System.out.println("LGTV.... 전원을 끈다");
	}
	@Override
	public void volumeUp() {
		System.out.println("LGTV.... 소리를 올린다");
	}
	@Override
	public void volumeDown() {
		System.out.println("LGTV.... 소리를 줄인다");
	}
}
