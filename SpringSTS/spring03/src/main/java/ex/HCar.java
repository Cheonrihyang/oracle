package ex;

public class HCar implements Car{

	Engine e;
	String name;
	
	public void setEngine(Engine e) {
		this.e = e;
	}
	public void setName(String name) {
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
		System.out.println("현대 자동차가 출발합니다.");
		
	}
	@Override
	public void stop() {
		System.out.println("현대 자동차가 멈춥니다.");
		
	}
	@Override
	public void print() {
		System.out.println("차종: "+name);
		
	}
}
