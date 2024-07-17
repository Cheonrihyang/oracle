package com.spring.view.controller;

import java.io.IOException;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;
import com.spring.biz.board.impl.BoardService;
import com.spring.biz.user.UserVO;
import com.spring.biz.user.impl.UserDAO;
import com.spring.biz.user.impl.UserService;
import com.spring.biz.user.impl.UserServiceImpl;

@WebServlet("*.do")
public class DispatcherServlet extends HttpServlet{
	
	AbstractApplicationContext container;
	BoardService boardService;
	UserService userService;
	
	@Override
	public void init() throws ServletException {
		container = new GenericXmlApplicationContext("applicationContext.xml");
		boardService = (BoardService)container.getBean("boardService");
		userService = (UserService)container.getBean("userService");
	}
	
	
	
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		req.setCharacterEncoding("utf-8");
		
		process(req, resp);
	}
	
	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		req.setCharacterEncoding("utf-8");
		
		process(req, resp);
	}
	
	protected void process(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		req.setCharacterEncoding("utf-8");
		
		String uri = req.getRequestURI();
		String path = uri.substring(uri.lastIndexOf("/"));
		
		if(path.equals("/login.do")) {
			System.out.println("login.do 성공");

			String id = req.getParameter("id");
			String password = req.getParameter("password");
			
			UserVO vo = new UserVO();
			vo.setId(id);
			vo.setPassword(password);

			
			//UserDAO dao = new UserDAO();
			UserVO user = userService.getUser(vo);
			
			if(user != null) {
				HttpSession session = req.getSession();
				session.setAttribute("ID", id);
				resp.sendRedirect("getBoardList.do");
			}else {
				resp.sendRedirect("login.jsp");
			}
			
			
		}else if(path.equals("/userCheck.do")) {
			System.out.println("userCheck.do 성공");
			
			String id = req.getParameter("id");
			String password = req.getParameter("password");
			
			UserVO userVO = new UserVO();
			userVO.setId(id);
			userVO.setPassword(password);
			
			//UserDAO userDAO = new UserDAO();
			UserVO user = userService.checkUserLogin(userVO);
			
			if(user != null) {
				HttpSession session = req.getSession();
				session.setAttribute("ID", id);
				resp.sendRedirect("getBoardList.do");
			}else {
				resp.sendRedirect("login.jsp");
			}
			
			
			
			
			
		}else if(path.equals("/getBoardList.do")) {
			System.out.println("getBoardList.do 성공");
			
			//BoardDAO boardDAO = new BoardDAO();
			List<BoardVO> boardList = boardService.getBoardList();
			
			HttpSession session = req.getSession();
			session.setAttribute("boardList", boardList);
			
			resp.sendRedirect("getBoardList.jsp");
			
			
		}else if(path.equals("/insertBoard.do")) {
			System.out.println("insertBoard.do 성공");
			
			BoardVO vo = new BoardVO();
			vo.setTitle(req.getParameter("title"));
			vo.setWriter(req.getParameter("writer"));
			vo.setContent(req.getParameter("content"));
			
			//BoardDAO boardDAO = new BoardDAO();
			boardService.insertBoard(vo);
			
			resp.sendRedirect("getBoardList.do");
			
			
		}else if(path.equals("/getBoard.do")) {
			System.out.println("getBoard.do 성공");
			
			String seq = req.getParameter("seq");
			
			BoardVO vo = new BoardVO();
			vo.setSeq(Integer.parseInt(seq));
			
			//BoardDAO boardDAO = new BoardDAO();
			BoardVO board = boardService.getBoard(vo);
			
			HttpSession session = req.getSession();
			session.setAttribute("board", board);
			resp.sendRedirect("getBoard.jsp");
			
			
		}else if(path.equals("/updateBoard.do")) {
			System.out.println("updateBoard.do 성공");
			
			String seq = req.getParameter("seq");
			String title = req.getParameter("title");
			String content = req.getParameter("content");
			
			BoardVO vo = new BoardVO();
			vo.setTitle(title);
			vo.setContent(content);
			vo.setSeq(Integer.parseInt(seq));
			
			//BoardDAO boardDAO = new BoardDAO();
			boardService.updateBoard(vo);
			
			resp.sendRedirect("getBoardList.do");
			
			
		}else if(path.equals("/deleteBoard.do")) {
			System.out.println("deleteBoard.do 성공");
			
			String seq = req.getParameter("seq");
			
			BoardVO vo = new BoardVO();
			vo.setSeq(Integer.parseInt(seq));
			
			BoardDAO boardService = new BoardDAO();
			boardService.deleteBoard(vo);
			
			resp.sendRedirect("getBoardList.do");
			
			
		}else if(path.equals("/logout.do")) {
			System.out.println("logout.do 성공");
			
			HttpSession session = req.getSession();
			session.invalidate();
			
			resp.sendRedirect("login.jsp");
		}
	}
}
