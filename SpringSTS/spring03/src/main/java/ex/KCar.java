package ex;

public class KCar implements Car{
	
	Engine e;
	String name;
	
	public KCar(Engine e, String name) {
		this.e = e;
		this.name = name;
	}

	@Override
	public void powerOn() {
		e.powerOn();
		
	}
	@Override
	public void powerOff() {
		e.powerOff();
		
	}
	@Override
	public void go() {
		System.out.println("기아 자동차가 출발합니다.");
		
	}
	@Override
	public void stop() {
		System.out.println("기아 자동차가 멈춥니다.");
		
	}

	@Override
	public void print() {
		System.out.println("차종: "+name);
		
	}
	
	
}
