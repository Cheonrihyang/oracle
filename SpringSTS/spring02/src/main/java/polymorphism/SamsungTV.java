package polymorphism;

public class SamsungTV implements TV{
	
	@Override
	public void powerOn() {
		System.out.println("SamsungTV....전원을 켠다");
	}
	@Override
	public void powerOff() {
		System.out.println("SamsungTV.... 전원을 끈다");
	}
	@Override
	public void volumeUp() {
		System.out.println("SamsungTV.... 소리를 올린다");
	}
	@Override
	public void volumeDown() {
		System.out.println("SamsungTV.... 소리를 줄인다");
	}
}
